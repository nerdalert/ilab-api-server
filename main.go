package main

import (
	"encoding/json"
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
	"github.com/spf13/cobra"
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

// ModelCache encapsulates the cached models and related metadata.
type ModelCache struct {
	Models []Model
	Time   time.Time
	Mutex  sync.Mutex
}

var (
	baseDir            string
	taxonomyPath       string
	rhelai             bool
	ilabCmd            string
	isOSX              bool
	isCuda             bool
	jobs               = make(map[string]*Job)
	jobsLock           = sync.Mutex{}
	modelLock          = sync.Mutex{}
	modelProcessBase   *exec.Cmd // Process for base model
	modelProcessLatest *exec.Cmd // Process for latest model
	baseModel          = "instructlab/granite-7b-lab"

	// Cache variables
	modelCache = ModelCache{}
)

const jobsFile = "jobs.json"

func main() {
	rootCmd := &cobra.Command{
		Use:   "ilab-server",
		Short: "ILab Server Application",
		Run:   runServer,
	}

	// Define flags
	rootCmd.Flags().BoolVar(&rhelai, "rhelai", false, "Use ilab binary from PATH instead of Python virtual environment")
	rootCmd.Flags().StringVar(&baseDir, "base-dir", "", "Base directory for ilab operations (required if --rhelai is not set)")
	rootCmd.Flags().StringVar(&taxonomyPath, "taxonomy-path", "", "Path to the taxonomy repository for Git operations (required)")
	rootCmd.Flags().BoolVar(&isOSX, "osx", false, "Enable OSX-specific settings (default: false)")
	rootCmd.Flags().BoolVar(&isCuda, "cuda", false, "Enable Cuda (default: false)")

	// Mark flags as required based on --rhelai
	rootCmd.PreRunE = func(cmd *cobra.Command, args []string) error {
		if !rhelai && baseDir == "" {
			return fmt.Errorf("--base-dir is required unless --rhelai is set")
		}
		if taxonomyPath == "" {
			return fmt.Errorf("--taxonomy-path is required")
		}
		return nil
	}

	if err := rootCmd.Execute(); err != nil {
		log.Fatalf("Error executing command: %v", err)
	}
}

func runServer(cmd *cobra.Command, args []string) {
	// Determine ilab command path
	if rhelai {
		// Use ilab from PATH
		ilabPath, err := exec.LookPath("ilab")
		if err != nil {
			log.Fatalf("ilab binary not found in PATH. Please ensure ilab is installed and in your PATH.")
		}
		ilabCmd = ilabPath
	} else {
		// Use ilab from virtual environment
		ilabCmd = filepath.Join(baseDir, "venv", "bin", "ilab")
		if _, err := os.Stat(ilabCmd); os.IsNotExist(err) {
			log.Fatalf("ilab binary not found at %s. Please ensure the virtual environment is set up correctly.", ilabCmd)
		}
	}

	log.Printf("Using ilab command: %s", ilabCmd)

	// Validate mandatory arguments if not using rhelai
	if !rhelai {
		if _, err := os.Stat(baseDir); os.IsNotExist(err) {
			log.Fatalf("Base directory does not exist: %s", baseDir)
		}
	}

	if _, err := os.Stat(taxonomyPath); os.IsNotExist(err) {
		log.Fatalf("Taxonomy path does not exist: %s", taxonomyPath)
	}

	log.Printf("Running with baseDir=%s, taxonomyPath=%s, isOSX=%v, isCuda=%v", baseDir, taxonomyPath, isOSX, isCuda)
	log.Printf("Current working directory: %s", mustGetCwd())

	// Load existing jobs from file
	loadJobs()

	// Check statuses of running jobs from previous sessions
	checkRunningJobs()

	// Initialize the model cache
	initializeModelCache()

	// Create the logs directory if it doesn't exist
	err := os.MkdirAll("logs", os.ModePerm)
	if err != nil {
		log.Fatalf("Failed to create logs directory: %v", err)
	}

	// Setup HTTP routes
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

	// Start the server with logging
	log.Printf("Server starting on port 8080... (Taxonomy path: %s)", taxonomyPath)
	if err := http.ListenAndServe("0.0.0.0:8080", r); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}

// sanitizeModelName checks if the modelName starts with "model/" and replaces it with "models/".
func sanitizeModelName(modelName string) string {
	if strings.HasPrefix(modelName, "model/") {
		return strings.Replace(modelName, "model/", "models/", 1)
	}
	return modelName
}

// mustGetCwd returns the current working directory or "unknown" if it fails.
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

// getIlabCommand returns the ilab command based on the --rhelai flag
func getIlabCommand() string {
	return ilabCmd
}

// getBaseCacheDir returns the base cache directory path: ~/.cache/instructlab/
func getBaseCacheDir() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %v", err)
	}
	return filepath.Join(homeDir, ".cache", "instructlab"), nil
}

// Helper function to construct the full model path: ~/.cache/instructlab/models/<modelName>
func getFullModelPath(modelName string) (string, error) {
	baseCacheDir, err := getBaseCacheDir()
	if err != nil {
		return "", err
	}
	// If modelName starts with "models/", do not prepend "models/" again.
	// Otherwise, prepend "models/".
	if strings.HasPrefix(modelName, "models/") {
		return filepath.Join(baseCacheDir, modelName), nil
	}
	return filepath.Join(baseCacheDir, "models", modelName), nil
}

// Helper function to get the latest dataset file
func getLatestDatasetFile() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %v", err)
	}
	datasetDir := filepath.Join(homeDir, ".local", "share", "instructlab", "datasets")
	files, err := ioutil.ReadDir(datasetDir)
	if err != nil {
		return "", fmt.Errorf("failed to read dataset directory: %v", err)
	}

	var latestFile os.FileInfo
	for _, file := range files {
		if strings.HasPrefix(file.Name(), "knowledge_train_msgs_") && strings.HasSuffix(file.Name(), ".jsonl") {
			if latestFile == nil || file.ModTime().After(latestFile.ModTime()) {
				latestFile = file
			}
		}
	}

	if latestFile == nil {
		return "", fmt.Errorf("no dataset file found with the prefix 'knowledge_train_msgs_'")
	}
	return filepath.Join(datasetDir, latestFile.Name()), nil
}

// Initialize the model cache on server startup and start periodic refresh
func initializeModelCache() {
	// Initial cache refresh
	refreshModelCache()

	// Start a goroutine to refresh the cache every 20 minutes
	go func() {
		for {
			time.Sleep(20 * time.Minute)
			refreshModelCache()
		}
	}()
}

// Refresh the model cache if it's older than 20 minutes
func refreshModelCache() {
	modelCache.Mutex.Lock()
	defer modelCache.Mutex.Unlock()

	// Check if cache is valid
	if time.Since(modelCache.Time) < 20*time.Minute && len(modelCache.Models) > 0 {
		log.Println("Model cache is still valid; no refresh needed.")
		return
	}

	log.Println("Refreshing model cache...")
	output, err := runIlabCommand("model", "list")
	if err != nil {
		log.Printf("Error refreshing model cache: %v", err)
		return
	}

	models, err := parseModelList(output)
	if err != nil {
		log.Printf("Error parsing model list during cache refresh: %v", err)
		return
	}

	modelCache.Models = models
	modelCache.Time = time.Now()
	log.Printf("Model cache refreshed at %v with %d models.", modelCache.Time, len(modelCache.Models))
}

// GetModels is the HTTP handler for the /models endpoint.
// It serves cached model data, refreshing the cache if necessary.
func getModels(w http.ResponseWriter, r *http.Request) {
	log.Println("GET /models called")

	// Lock the cache for reading
	modelCache.Mutex.Lock()
	cachedTime := modelCache.Time
	cachedModels := make([]Model, len(modelCache.Models))
	copy(cachedModels, modelCache.Models)
	modelCache.Mutex.Unlock()

	// Check if cache is valid
	if len(cachedModels) > 0 && time.Since(cachedTime) < 20*time.Minute {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(cachedModels); err != nil {
			log.Printf("Error encoding cached models: %v", err)
			http.Error(w, "Failed to encode models", http.StatusInternalServerError)
			return
		}
		log.Println("GET /models returned cached models.")
		return
	}

	// If cache is empty or stale, refresh the cache
	log.Println("Cache is empty or stale. Refreshing model cache, blocking until complete ~15s...")
	refreshModelCache()

	// After refresh, attempt to serve the cache
	modelCache.Mutex.Lock()
	cachedTime = modelCache.Time
	cachedModels = make([]Model, len(modelCache.Models))
	copy(cachedModels, modelCache.Models)
	modelCache.Mutex.Unlock()

	if len(cachedModels) > 0 {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(cachedModels); err != nil {
			log.Printf("Error encoding refreshed models: %v", err)
			http.Error(w, "Failed to encode models", http.StatusInternalServerError)
			return
		}
		log.Println("GET /models returned refreshed models.")
	} else {
		http.Error(w, "Failed to retrieve models", http.StatusInternalServerError)
		log.Println("GET /models failed to retrieve models.")
	}
}

// runIlabCommand executes the ilab command with the provided arguments.
func runIlabCommand(args ...string) (string, error) {
	cmdPath := getIlabCommand()
	cmd := exec.Command(cmdPath, args...)
	if !rhelai {
		cmd.Dir = baseDir
	}
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// parseModelList parses the output of the "ilab model list" command into a slice of Model structs.
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
			model := Model{
				Name:         strings.TrimSpace(fields[0]),
				LastModified: strings.TrimSpace(fields[1]),
				Size:         strings.TrimSpace(fields[2]),
			}
			models = append(models, model)
		}
	}
	return models, nil
}

// getData is the HTTP handler for the /data endpoint.
func getData(w http.ResponseWriter, r *http.Request) {
	log.Println("GET /data called")
	output, err := runIlabCommand("data", "list")
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

// parseDataList parses the output of the "ilab data list" command into a slice of Data structs.
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
			data := Data{
				Dataset:   strings.TrimSpace(fields[0]),
				CreatedAt: strings.TrimSpace(fields[1]),
				FileSize:  strings.TrimSpace(fields[2]),
			}
			dataList = append(dataList, data)
		}
	}
	return dataList, nil
}

// generateData is the HTTP handler for the /data/generate endpoint.
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

// startGenerateJob starts the data generation job and returns the job ID.
func startGenerateJob() (string, error) {
	ilabPath := getIlabCommand()
	cmdArgs := []string{"data", "generate"}
	if rhelai {
		cmdArgs = append(cmdArgs, "--pipeline", "full")
	} else {
		cmdArgs = append(cmdArgs, "--pipeline", "simple")
	}
	cmd := exec.Command(ilabPath, cmdArgs...)

	if !rhelai {
		cmd.Dir = baseDir
	}

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

	log.Printf("Running command: %s %v", ilabPath, cmdArgs)
	if err := cmd.Start(); err != nil {
		log.Printf("Error starting data generation command: %v", err)
		logFile.Close()
		return "", err
	}

	job := &Job{
		JobID:     jobID,
		Cmd:       ilabPath,
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
			log.Printf("Job %s failed with error: %v", job.JobID, err)
		} else {
			if cmd.ProcessState.Success() {
				job.Status = "finished"
				log.Printf("Job %s finished successfully", job.JobID)
			} else {
				job.Status = "failed"
				log.Printf("Job %s failed", job.JobID)
			}
		}

		now := time.Now()
		job.EndTime = &now
		saveJobs()
	}()

	return jobID, nil
}

// trainModel is the HTTP handler for the /model/train endpoint.
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

	// Sanitize the modelName
	sanitizedModelName := sanitizeModelName(reqBody.ModelName)
	log.Printf("Sanitized modelName: '%s'", sanitizedModelName)

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
	jobID, err := startTrainJob(sanitizedModelName, reqBody.BranchName)
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

// startTrainJob starts the training job and returns the job ID.
func startTrainJob(modelName, branchName string) (string, error) {
	log.Printf("Starting training job for model: '%s', branch: '%s'", modelName, branchName)

	// Generate unique job ID
	jobID := fmt.Sprintf("t-%d", time.Now().UnixNano())
	logFilePath := filepath.Join("logs", fmt.Sprintf("%s.log", jobID))

	// Get the full model path
	fullModelPath, err := getFullModelPath(modelName)
	if err != nil {
		return "", fmt.Errorf("failed to get full model path: %v", err)
	}

	// Ensure the model directory exists
	modelDir := filepath.Dir(fullModelPath)
	err = os.MkdirAll(modelDir, os.ModePerm)
	if err != nil {
		return "", fmt.Errorf("failed to create model directory '%s': %v", modelDir, err)
	}

	// Training options
	ilabPath := getIlabCommand()
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

	// Check if RHELAI is enabled
	if rhelai {
		// Train with the most recent dataset
		latestDataset, err := getLatestDatasetFile()
		if err != nil {
			return "", fmt.Errorf("failed to get latest dataset file: %v", err)
		}
		cmdArgs = []string{
			"model", "train",
			"--pipeline=accelerated",
			fmt.Sprintf("--data-path=%s", latestDataset),
			"--max-batch-len=10000",
			"--gpus=4",
			"--device=cuda",
			"--save-samples=0",
			"--distributed-backend=fsdp",
			fmt.Sprintf("--model-path=%s", fullModelPath),
		}
	}

	cmd := exec.Command(ilabPath, cmdArgs...)
	if !rhelai {
		cmd.Dir = baseDir
	}

	log.Printf("Training command: %s %v", ilabPath, cmdArgs)
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
		Cmd:       ilabPath,
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
			log.Printf("Training job '%s' failed: %v", job.JobID, err)
		} else if cmd.ProcessState.Success() {
			job.Status = "finished"
			log.Printf("Training job '%s' finished successfully", job.JobID)
		} else {
			job.Status = "failed"
			log.Printf("Training job '%s' failed (unknown reason)", job.JobID)
		}

		now := time.Now()
		job.EndTime = &now
		saveJobs()
	}()

	return jobID, nil
}

// getJobStatus is the HTTP handler for the /jobs/{job_id}/status endpoint.
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

// listJobs is the HTTP handler for the /jobs endpoint.
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

// getJobLogs is the HTTP handler for the /jobs/{job_id}/logs endpoint.
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

// generateTrainPipeline is the HTTP handler for the /pipeline/generate-train endpoint.
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

	// Sanitize the modelName
	sanitizedModelName := sanitizeModelName(reqBody.ModelName)
	log.Printf("Sanitized modelName for pipeline: '%s'", sanitizedModelName)

	// Create a unique pipeline job ID
	pipelineJobID := fmt.Sprintf("p-%d", time.Now().UnixNano())
	log.Printf("Starting pipeline job with ID: %s", pipelineJobID)

	// Save the pipeline job as a placeholder
	job := &Job{
		JobID:     pipelineJobID,
		Cmd:       "pipeline-generate-train",
		Args:      []string{sanitizedModelName, reqBody.BranchName},
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
	go runPipelineJob(job, sanitizedModelName, reqBody.BranchName)

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

// serveModel starts serving a model on the specified port.
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

	var cmdArgs []string
	cmdArgs = []string{
		"serve", "model",
		"--model", modelPath,
		"--host", "0.0.0.0",
		"--port", port,
	}

	cmdPath := getIlabCommand()
	cmd := exec.Command(cmdPath, cmdArgs...)
	if !rhelai {
		cmd.Dir = baseDir
	}

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
		Cmd:       cmdPath,
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

	// Monitor the model process
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

// serveLatestCheckpoint serves the latest checkpoint model on port 8001.
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

// serveBaseModel serves the "base" model on port 8000.
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

// runPipelineJob executes the pipeline steps: Git checkout, data generation, and training.
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

	// Perform Git checkout
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

	// Start data generation step
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

	// Wait for data generation to finish
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

	// Start training step
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

	// Wait for training to finish
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

	// Pipeline completed successfully
	jobsLock.Lock()
	job.Status = "finished"
	jobsLock.Unlock()
	saveJobs()
	logger.Println("Pipeline job completed successfully.")
}
