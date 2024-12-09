package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/mux"
)

type Model struct {
	Name         string `json:"name"`
	LastModified string `json:"last_modified"`
	Size         string `json:"size"`
}

type Data struct {
	Dataset   string `json:"dataset"`
	CreatedAt string `json:"created_at"`
	FileSize  string `json:"file_size"`
}

type Job struct {
	JobID     string     `json:"job_id"`
	Cmd       string     `json:"cmd"`
	Args      []string   `json:"args"`
	Status    string     `json:"status"` // "running", "finished", "failed"
	PID       int        `json:"pid"`
	LogFile   string     `json:"log_file"`
	StartTime time.Time  `json:"start_time"`
	EndTime   *time.Time `json:"end_time,omitempty"`
	Branch    string     `json:"branch"`
	Lock      sync.Mutex `json:"-"`
}

var (
	baseDir            string
	taxonomyPath       string
	jobs               = make(map[string]*Job)
	jobsLock           = sync.Mutex{}
	baseModel          = "instructlab/granite-7b-lab"
	isOSX              bool
	isCuda             bool
	modelLock          = sync.Mutex{}
	modelProcessBase   *exec.Cmd // Process for base model
	modelProcessLatest *exec.Cmd // Process for latest model
)

const jobsFile = "jobs.json"

func main() {
	// Define mandatory flags
	flag.StringVar(&baseDir, "base-dir", "", "Base directory for ilab operations (required)")
	flag.StringVar(&taxonomyPath, "taxonomy-path", "", "Path to the taxonomy repository for Git operations (required)")
	osx := flag.Bool("osx", false, "Enable OSX-specific settings (default: false)")
	cuda := flag.Bool("cuda", false, "Enable Cuda (default: false)")
	flag.Parse()

	// Validate mandatory arguments
	if baseDir == "" || taxonomyPath == "" {
		log.Fatalf("Both --base-dir and --taxonomy-path must be specified")
	}

	// Validate that the directories exist
	if _, err := os.Stat(baseDir); os.IsNotExist(err) {
		log.Fatalf("Base directory does not exist: %s", baseDir)
	}
	if _, err := os.Stat(taxonomyPath); os.IsNotExist(err) {
		log.Fatalf("Taxonomy path does not exist: %s", taxonomyPath)
	}

	isOSX = *osx
	isCuda = *cuda

	log.Printf("Running with baseDir=%s, taxonomyPath=%s, isOSX=%v, isCuda=%v", baseDir, taxonomyPath, isOSX, isCuda)
	log.Printf("Current working directory: %s", mustGetCwd())

	// Load existing jobs from file
	loadJobs()

	// Check statuses of running jobs from previous sessions
	checkRunningJobs()

	r := mux.NewRouter()
	r.HandleFunc("/models", getModels).Methods("GET")
	r.HandleFunc("/data", getData).Methods("GET")
	r.HandleFunc("/data/generate", generateData).Methods("POST")
	r.HandleFunc("/model/train", trainModel).Methods("POST")
	r.HandleFunc("/jobs/{job_id}/status", getJobStatus).Methods("GET")
	r.HandleFunc("/jobs/{job_id}/logs", getJobLogs).Methods("GET")
	r.HandleFunc("/jobs", listJobs).Methods("GET")
	r.HandleFunc("/pipeline/generate-train", generateTrainPipeline).Methods("POST")
	r.HandleFunc("/model/serve-latest", serveLatestCheckpoint).Methods("POST")
	r.HandleFunc("/model/serve-base", serveBaseModel).Methods("POST")

	// Create the logs directory if it doesn't exist
	err := os.MkdirAll("logs", os.ModePerm)
	if err != nil {
		log.Fatalf("Failed to create logs directory: %v", err)
	}

	// Start the server with logging
	log.Printf("Server starting on port 8080... (Base directory: %s, Taxonomy path: %s)", baseDir, taxonomyPath)
	if err := http.ListenAndServe("0.0.0.0:8080", r); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}

func mustGetCwd() string {
	cwd, err := os.Getwd()
	if err != nil {
		return "unknown"
	}
	return cwd
}

// Load jobs from the jobs.json file
func loadJobs() {
	jobsLock.Lock()
	defer jobsLock.Unlock()

	if _, err := os.Stat(jobsFile); os.IsNotExist(err) {
		// No jobs file exists
		return
	}

	data, err := ioutil.ReadFile(jobsFile)
	if err != nil {
		log.Printf("Error reading jobs file: %v", err)
		return
	}

	err = json.Unmarshal(data, &jobs)
	if err != nil {
		log.Printf("Error unmarshalling jobs data: %v", err)
		return
	}

	log.Printf("Loaded %d jobs from %s", len(jobs), jobsFile)
}

// Save jobs to the jobs.json file
func saveJobs() {
	jobsLock.Lock()
	defer jobsLock.Unlock()

	data, err := json.MarshalIndent(jobs, "", "  ")
	if err != nil {
		log.Printf("Error marshalling jobs data: %v", err)
		return
	}

	err = ioutil.WriteFile(jobsFile, data, 0644)
	if err != nil {
		log.Printf("Error writing jobs file: %v", err)
	}
}

// Check the status of running jobs after server restart
func checkRunningJobs() {
	jobsLock.Lock()
	changed := false
	for _, job := range jobs {
		if job.Status == "running" {
			// Check if the process is still running
			processRunning := isProcessRunning(job.PID)
			if !processRunning {
				job.Status = "failed"
				changed = true
				log.Printf("Job %s marked as failed (process not running)", job.JobID)
			}
		}
	}
	jobsLock.Unlock()

	if changed {
		saveJobs()
	}
}

// Check if a process with the given PID is running
func isProcessRunning(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	err = process.Signal(syscall.Signal(0))
	return err == nil
}

func runInVirtualEnv(args ...string) (string, error) {
	cmd := exec.Command("./venv/bin/ilab", args...)
	cmd.Dir = baseDir
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func getModels(w http.ResponseWriter, r *http.Request) {
	log.Println("GET /models called")
	output, err := runInVirtualEnv("model", "list")
	if err != nil {
		log.Printf("Error running 'ilab model list': %v", err)
		http.Error(w, string(output), http.StatusInternalServerError)
		return
	}
	models, err := parseModelList(output)
	if err != nil {
		log.Printf("Error parsing model list: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
	log.Println("GET /models successful")
}

func parseModelList(output string) ([]Model, error) {
	var models []Model
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "+") || strings.HasPrefix(line, "| Model Name") || line == "" {
			continue
		}
		if strings.HasPrefix(line, "|") {
			line = strings.Trim(line, "|")
			fields := strings.Split(line, "|")
			if len(fields) != 3 {
				continue
			}
			name := strings.TrimSpace(fields[0])
			lastModified := strings.TrimSpace(fields[1])
			size := strings.TrimSpace(fields[2])
			model := Model{
				Name:         name,
				LastModified: lastModified,
				Size:         size,
			}
			models = append(models, model)
		}
	}
	return models, nil
}

func getData(w http.ResponseWriter, r *http.Request) {
	log.Println("GET /data called")
	output, err := runInVirtualEnv("data", "list")
	if err != nil {
		log.Printf("Error running 'ilab data list': %v", err)
		http.Error(w, string(output), http.StatusInternalServerError)
		return
	}
	dataList, err := parseDataList(output)
	if err != nil {
		log.Printf("Error parsing data list: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(dataList)
	log.Println("GET /data successful")
}

func parseDataList(output string) ([]Data, error) {
	var dataList []Data
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "+") || strings.HasPrefix(line, "| Dataset") || line == "" {
			continue
		}
		if strings.HasPrefix(line, "|") {
			line = strings.Trim(line, "|")
			fields := strings.Split(line, "|")
			if len(fields) != 3 {
				continue
			}
			dataset := strings.TrimSpace(fields[0])
			createdAt := strings.TrimSpace(fields[1])
			fileSize := strings.TrimSpace(fields[2])
			data := Data{
				Dataset:   dataset,
				CreatedAt: createdAt,
				FileSize:  fileSize,
			}
			dataList = append(dataList, data)
		}
	}
	return dataList, nil
}

func generateData(w http.ResponseWriter, r *http.Request) {
	log.Println("POST /data/generate called")
	jobID, err := startGenerateJob()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"job_id": jobID})
	log.Printf("POST /data/generate successful, job_id: %s", jobID)
}

func startGenerateJob() (string, error) {
	cmdPath := "./venv/bin/ilab"
	cmdArgs := []string{"data", "generate", "--pipeline", "simple"}
	cmd := exec.Command(cmdPath, cmdArgs...)
	cmd.Dir = baseDir

	jobID := fmt.Sprintf("g-%d", time.Now().UnixNano())
	logFilePath := filepath.Join("logs", fmt.Sprintf("%s.log", jobID))
	log.Printf("Starting generateData job: %s, logs: %s", jobID, logFilePath)
	logFile, err := os.Create(logFilePath)
	if err != nil {
		log.Printf("Error creating log file: %v", err)
		return "", fmt.Errorf("Failed to create log file")
	}

	cmd.Stdout = logFile
	cmd.Stderr = logFile

	log.Printf("Running command: %s %v", cmdPath, cmdArgs)
	if err := cmd.Start(); err != nil {
		log.Printf("Error starting data generation command: %v", err)
		logFile.Close()
		return "", err
	}

	job := &Job{
		JobID:     jobID,
		Cmd:       cmdPath,
		Args:      cmdArgs,
		Status:    "running",
		PID:       cmd.Process.Pid,
		LogFile:   logFilePath,
		StartTime: time.Now(),
	}

	jobsLock.Lock()
	jobs[jobID] = job
	jobsLock.Unlock()

	saveJobs()

	go func() {
		err := cmd.Wait()
		logFile.Close()

		job.Lock.Lock()
		defer job.Lock.Unlock()

		if err != nil {
			job.Status = "failed"
			log.Printf("Job %s failed with error: %v", jobID, err)
		} else {
			if cmd.ProcessState.Success() {
				job.Status = "finished"
				log.Printf("Job %s finished successfully", jobID)
			} else {
				job.Status = "failed"
				log.Printf("Job %s failed", jobID)
			}
		}

		now := time.Now()
		job.EndTime = &now
		saveJobs()
	}()

	return jobID, nil
}

func trainModel(w http.ResponseWriter, r *http.Request) {
	log.Println("POST /model/train called")

	var reqBody struct {
		ModelName  string `json:"modelName"`
		BranchName string `json:"branchName"`
	}

	// Parse the request body
	if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
		log.Printf("Error parsing request body: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	log.Printf("Received train request with modelName: '%s', branchName: '%s'", reqBody.ModelName, reqBody.BranchName)

	// Ensure required fields are provided
	if reqBody.ModelName == "" || reqBody.BranchName == "" {
		log.Println("Missing required parameters: modelName or branchName")
		http.Error(w, "Missing required parameters: modelName or branchName", http.StatusBadRequest)
		return
	}

	// Perform Git checkout
	gitCheckoutCmd := exec.Command("git", "checkout", reqBody.BranchName)
	gitCheckoutCmd.Dir = taxonomyPath
	gitOutput, err := gitCheckoutCmd.CombinedOutput()

	log.Printf("Git checkout output: %s", string(gitOutput))

	if err != nil {
		log.Printf("Error checking out branch '%s': %v", reqBody.BranchName, err)
		http.Error(w, fmt.Sprintf("Failed to checkout branch '%s': %s", reqBody.BranchName, string(gitOutput)), http.StatusInternalServerError)
		return
	}

	log.Printf("Successfully checked out branch: '%s'", reqBody.BranchName)

	// Start the training job
	jobID, err := startTrainJob(baseModel, reqBody.BranchName)
	if err != nil {
		log.Printf("Error starting train job: %v", err)
		http.Error(w, "Failed to start train job", http.StatusInternalServerError)
		return
	}

	log.Printf("Train job started successfully with job_id: '%s'", jobID)

	// Return the job ID in the response
	response := map[string]string{
		"job_id": jobID,
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding response: %v", err)
		http.Error(w, "Failed to send response", http.StatusInternalServerError)
		return
	}

	log.Println("POST /model/train response sent successfully")
}

func startTrainJob(modelName, branchName string) (string, error) {
	log.Printf("Starting training job for model: '%s', branch: '%s'", modelName, branchName)

	// Generate unique job ID
	jobID := fmt.Sprintf("t-%d", time.Now().UnixNano())
	logFilePath := filepath.Join("logs", fmt.Sprintf("%s.log", jobID))

	//  training opts
	cmdPath := "./venv/bin/ilab"
	cmdArgs := []string{
		"model", "train",
		"--pipeline=simple",
		fmt.Sprintf("--model-path=%s", modelName),
	}
	if isOSX {
		cmdArgs = append(cmdArgs, "--device=mps")
	}
	if isCuda {
		cmdArgs = append(cmdArgs, "--device=cuda")
	}

	cmd := exec.Command(cmdPath, cmdArgs...)
	cmd.Dir = baseDir

	log.Printf("Training command: %s %v", cmdPath, cmdArgs)
	logFile, err := os.Create(logFilePath)
	if err != nil {
		log.Printf("Error creating log file: %v", err)
		return "", fmt.Errorf("Failed to create log file: %v", err)
	}
	defer logFile.Close()

	// Redirect command output to log file
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	// Start the command
	if err := cmd.Start(); err != nil {
		log.Printf("Error starting training command: %v", err)
		return "", err
	}

	log.Printf("Training process started with PID: %d", cmd.Process.Pid)

	// Save job details
	job := &Job{
		JobID:     jobID,
		Cmd:       cmdPath,
		Args:      cmdArgs,
		Status:    "running",
		PID:       cmd.Process.Pid,
		LogFile:   logFilePath,
		Branch:    branchName,
		StartTime: time.Now(),
	}

	jobsLock.Lock()
	jobs[jobID] = job
	jobsLock.Unlock()
	saveJobs()

	// Wait for process completion in a goroutine
	go func() {
		err := cmd.Wait()

		job.Lock.Lock()
		defer job.Lock.Unlock()

		if err != nil {
			job.Status = "failed"
			log.Printf("Training job '%s' failed: %v", jobID, err)
		} else if cmd.ProcessState.Success() {
			job.Status = "finished"
			log.Printf("Training job '%s' finished successfully", jobID)
		} else {
			job.Status = "failed"
			log.Printf("Training job '%s' failed (unknown reason)", jobID)
		}

		saveJobs()
	}()

	return jobID, nil
}

func getJobStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["job_id"]
	log.Printf("GET /jobs/%s/status called", jobID)
	jobsLock.Lock()
	job, exists := jobs[jobID]
	jobsLock.Unlock()
	if !exists {
		log.Printf("Job %s not found", jobID)
		http.Error(w, "Job not found", http.StatusNotFound)
		return
	}
	job.Lock.Lock()
	status := job.Status
	job.Lock.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"job_id":  job.JobID,
		"status":  job.Status,
		"branch":  job.Branch,
		"command": job.Cmd,
	})
	log.Printf("GET /jobs/%s/status successful, status: %s", jobID, status)
}

func listJobs(w http.ResponseWriter, r *http.Request) {
	log.Println("GET /jobs called")
	jobsLock.Lock()
	defer jobsLock.Unlock()
	var jobList []Job
	for _, job := range jobs {
		job.Lock.Lock()
		jobCopy := *job
		job.Lock.Unlock()
		jobList = append(jobList, jobCopy)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(jobList)
}

func getJobLogs(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["job_id"]
	log.Printf("GET /jobs/%s/logs called", jobID)

	jobsLock.Lock()
	job, exists := jobs[jobID]
	jobsLock.Unlock()

	if !exists {
		log.Printf("Job %s not found", jobID)
		http.Error(w, "Job not found", http.StatusNotFound)
		return
	}

	if _, err := os.Stat(job.LogFile); os.IsNotExist(err) {
		log.Printf("Log file for job %s not found", jobID)
		http.Error(w, "Log file not found", http.StatusNotFound)
		return
	}

	logContent, err := ioutil.ReadFile(job.LogFile)
	if err != nil {
		log.Printf("Error reading log file for job %s: %v", jobID, err)
		http.Error(w, "Failed to read log file", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/plain")
	w.Write(logContent)
	log.Printf("GET /jobs/%s/logs successful", jobID)
}

func generateTrainPipeline(w http.ResponseWriter, r *http.Request) {
	log.Println("POST /pipeline/generate-train called")
	var reqBody struct {
		ModelName  string `json:"modelName"`
		BranchName string `json:"branchName"`
	}
	if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
		log.Printf("Error parsing request body: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Ensure required fields are provided
	if reqBody.ModelName == "" || reqBody.BranchName == "" {
		log.Println("Missing required parameters: modelName or branchName")
		http.Error(w, "Missing required parameters: modelName or branchName", http.StatusBadRequest)
		return
	}

	// Create a unique pipeline job ID
	pipelineJobID := fmt.Sprintf("p-%d", time.Now().UnixNano())
	log.Printf("Starting pipeline job with ID: %s", pipelineJobID)

	// Save the pipeline job as a placeholder
	job := &Job{
		JobID:     pipelineJobID,
		Cmd:       "pipeline-generate-train",
		Args:      []string{reqBody.ModelName, reqBody.BranchName},
		Status:    "running",
		PID:       0,
		LogFile:   fmt.Sprintf("logs/%s.log", pipelineJobID),
		Branch:    reqBody.BranchName,
		StartTime: time.Now(),
	}

	jobsLock.Lock()
	jobs[pipelineJobID] = job
	jobsLock.Unlock()

	saveJobs()

	// Start the pipeline in a separate goroutine
	go runPipelineJob(job, reqBody.ModelName, reqBody.BranchName)

	// Respond immediately with the pipeline job ID
	response := map[string]string{
		"pipeline_job_id": pipelineJobID,
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding response: %v", err)
		http.Error(w, "Failed to send response", http.StatusInternalServerError)
		return
	}

	log.Printf("POST /pipeline/generate-train response sent successfully with job_id: %s", pipelineJobID)
}

func runPipelineJob(job *Job, modelName, branchName string) {
	logFile, err := os.Create(job.LogFile)
	if err != nil {
		log.Printf("Error creating pipeline log file for job %s: %v", job.JobID, err)
		jobsLock.Lock()
		job.Status = "failed"
		jobsLock.Unlock()
		saveJobs()
		return
	}
	defer logFile.Close()

	logger := log.New(logFile, "", log.LstdFlags)

	logger.Printf("Starting pipeline job: %s, model: %s, branch: %s", job.JobID, modelName, branchName)

	gitCheckoutCmd := exec.Command("git", "checkout", branchName)
	gitCheckoutCmd.Dir = taxonomyPath
	gitOutput, gitErr := gitCheckoutCmd.CombinedOutput()
	logger.Printf("Git checkout output: %s", string(gitOutput))
	if gitErr != nil {
		logger.Printf("Failed to checkout branch '%s': %v", branchName, gitErr)
		jobsLock.Lock()
		job.Status = "failed"
		jobsLock.Unlock()
		saveJobs()
		return
	}

	logger.Println("Starting data generation step...")
	genJobID, genErr := startGenerateJob()
	if genErr != nil {
		logger.Printf("Data generation step failed: %v", genErr)
		jobsLock.Lock()
		job.Status = "failed"
		jobsLock.Unlock()
		saveJobs()
		return
	}
	logger.Printf("Data generation step started successfully with job_id: '%s'", genJobID)

	for {
		time.Sleep(5 * time.Second)
		jobsLock.Lock()
		genJob, exists := jobs[genJobID]
		jobsLock.Unlock()

		if !exists || genJob.Status == "failed" {
			logger.Println("Data generation step failed.")
			jobsLock.Lock()
			job.Status = "failed"
			jobsLock.Unlock()
			saveJobs()
			return
		}

		if genJob.Status == "finished" {
			logger.Println("Data generation step completed successfully.")
			break
		}
	}

	logger.Println("Starting training step...")
	trainJobID, trainErr := startTrainJob(modelName, branchName)
	if trainErr != nil {
		logger.Printf("Training step failed: %v", trainErr)
		jobsLock.Lock()
		job.Status = "failed"
		jobsLock.Unlock()
		saveJobs()
		return
	}
	logger.Printf("Training step started successfully with job_id: '%s'", trainJobID)

	for {
		time.Sleep(5 * time.Second)
		jobsLock.Lock()
		tJob, tExists := jobs[trainJobID]
		jobsLock.Unlock()

		if !tExists || tJob.Status == "failed" {
			logger.Println("Training step failed.")
			jobsLock.Lock()
			job.Status = "failed"
			jobsLock.Unlock()
			saveJobs()
			return
		}

		if tJob.Status == "finished" {
			logger.Println("Training step completed successfully.")
			break
		}
	}

	jobsLock.Lock()
	job.Status = "finished"
	jobsLock.Unlock()
	saveJobs()
	logger.Println("Pipeline job completed successfully.")
}

func serveModel(modelPath, port string, w http.ResponseWriter) {
	modelLock.Lock()
	defer modelLock.Unlock()

	log.Printf("serveModel called with modelPath=%s, port=%s", modelPath, port)

	// Determine which model we are serving based on port
	var targetProcess **exec.Cmd
	if port == "8000" {
		targetProcess = &modelProcessBase
	} else if port == "8001" {
		targetProcess = &modelProcessLatest
	} else {
		http.Error(w, "Invalid port specified", http.StatusBadRequest)
		return
	}

	// Check model file existence
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Printf("Model path does not exist: %s", modelPath)
		http.Error(w, fmt.Sprintf("Model path does not exist: %s", modelPath), http.StatusNotFound)
		return
	}
	log.Printf("Model file found at: %s", modelPath)

	// Kill only the process corresponding to this port
	if *targetProcess != nil && (*targetProcess).Process != nil {
		log.Printf("Stopping existing model process on port %s...", port)
		if err := (*targetProcess).Process.Kill(); err != nil {
			log.Printf("Failed to kill existing model process on port %s: %v", port, err)
			http.Error(w, "Failed to stop existing model process", http.StatusInternalServerError)
			return
		}
		*targetProcess = nil
	}

	pythonPath := filepath.Join(baseDir, "venv", "bin", "python")
	if _, err := os.Stat(pythonPath); os.IsNotExist(err) {
		log.Printf("Python binary does not exist at %s", pythonPath)
		http.Error(w, "Python binary does not exist. Check virtualenv setup.", http.StatusInternalServerError)
		return
	}

	cmdArgs := []string{
		"-m", "llama_cpp.server",
		"--model", modelPath,
		"--host", "0.0.0.0",
		"--port", port,
	}
	if !isOSX {
		cmdArgs = append(cmdArgs, "--n_gpu_layers", "-1")
	}

	log.Printf("Starting model serve with: %s %v", pythonPath, cmdArgs)
	cmd := exec.Command(pythonPath, cmdArgs...)
	cmd.Dir = baseDir

	jobID := fmt.Sprintf("ml-%d", time.Now().UnixNano())
	logFilePath := filepath.Join("logs", fmt.Sprintf("%s.log", jobID))
	log.Printf("Model serve logs: %s", logFilePath)
	logFile, err := os.Create(logFilePath)
	if err != nil {
		log.Printf("Error creating model run log file: %v", err)
		http.Error(w, "Failed to create log file", http.StatusInternalServerError)
		return
	}

	cmd.Stdout = logFile
	cmd.Stderr = logFile

	log.Println("Attempting to start model process...")
	if err := cmd.Start(); err != nil {
		log.Printf("Error starting model process: %v", err)
		logFile.Close()
		http.Error(w, "Failed to start model process", http.StatusInternalServerError)
		return
	}

	*targetProcess = cmd
	log.Printf("Model process started with PID %d on port %s", cmd.Process.Pid, port)

	// Save job details
	job := &Job{
		JobID:     jobID,
		Cmd:       pythonPath,
		Args:      cmdArgs,
		Status:    "running",
		PID:       cmd.Process.Pid,
		LogFile:   logFilePath,
		StartTime: time.Now(),
	}
	log.Printf("Model serve job details: %+v", job)

	jobsLock.Lock()
	jobs[jobID] = job
	jobsLock.Unlock()
	saveJobs()

	go func() {
		log.Printf("Waiting for model process to finish (job_id: %s, port: %s)", jobID, port)
		err := cmd.Wait()
		logFile.Sync()
		logFile.Close()

		job.Lock.Lock()
		defer job.Lock.Unlock()

		if err != nil {
			job.Status = "failed"
			log.Printf("Model run job '%s' on port %s failed: %v", jobID, port, err)
		} else if cmd.ProcessState.Success() {
			job.Status = "finished"
			log.Printf("Model run job '%s' on port %s finished successfully", jobID, port)
		} else {
			job.Status = "failed"
			log.Printf("Model run job '%s' on port %s failed (unknown reason)", jobID, port)
		}

		now := time.Now()
		job.EndTime = &now
		saveJobs()

		// If the process ends, clear the reference
		modelLock.Lock()
		defer modelLock.Unlock()
		if port == "8000" {
			if modelProcessBase == cmd {
				modelProcessBase = nil
			}
		} else if port == "8001" {
			if modelProcessLatest == cmd {
				modelProcessLatest = nil
			}
		}
	}()

	log.Printf("Model serve started successfully on port %s, returning job_id: %s", port, jobID)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "model process started", "job_id": jobID})
}

// serveLatestCheckpoint uses the helper function to serve the latest checkpoint model on port 8001
func serveLatestCheckpoint(w http.ResponseWriter, r *http.Request) {
	log.Println("POST /model/serve-latest called, loading the latest checkpoint")

	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Printf("Error getting user home directory: %v", err)
		http.Error(w, "Failed to get home directory", http.StatusInternalServerError)
		return
	}

	latestModelPath := filepath.Join(homeDir, ".local", "share", "instructlab", "checkpoints", "ggml-model-f16.gguf")
	log.Printf("Serving latest model at %s on port 8001", latestModelPath)
	serveModel(latestModelPath, "8001", w)
}

// serveBaseModel uses the helper function to serve the "base" model on port 8000
func serveBaseModel(w http.ResponseWriter, r *http.Request) {
	log.Println("POST /model/serve-base called")

	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Printf("Error getting user home directory: %v", err)
		http.Error(w, "Failed to get home directory", http.StatusInternalServerError)
		return
	}

	baseModelPath := filepath.Join(homeDir, ".cache", "instructlab", "models", "granite-7b-lab-Q4_K_M.gguf")
	log.Printf("Serving base model at %s on port 8000", baseModelPath)
	serveModel(baseModelPath, "8000", w)
}
