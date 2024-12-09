# Ilab API Server

## Overview

This is an Ilab API Server that is a temporary set of APIs for service developing apps against [InstructLab](https://github.com/instructlab/). It provides endpoints for model management, data generation, training, job tracking and job logging.

## Quickstart

### Prerequisites

- Ensure that the required directories (`base-dir` and `taxonomy-path`) exist and are accessible and Go is installed in the $PATH.

### Install Dependencies

To install the necessary dependencies, run:

```bash
go mod download
```

### Run the Server

#### For macOS with Metal (MPS):
```bash
go run main.go -base-dir /path/to/base-dir -taxonomy-path /path/to/taxonomy -osx
```

#### For CUDA-enabled environments:
```bash
go run main.go -base-dir /path/to/base-dir -taxonomy-path /path/to/taxonomy -cuda
```

Replace `/path/to/base-dir` and `/path/to/taxonomy` with your actual directories.

### Example:
```bash
go run main.go -base-dir /Users/user/code/instructlab -taxonomy-path /Users/user/code/taxonomy -osx
```

## API Doc

### Models

#### Get Models
**Endpoint**: `GET /models`  
Fetches the list of available models.

- **Response**:
  ```json
  [
    {
      "name": "model-name",
      "last_modified": "timestamp",
      "size": "size-string"
    }
  ]
  ```

### Data

#### Get Data
**Endpoint**: `GET /data`  
Fetches the list of datasets.

- **Response**:
  ```json
  [
    {
      "dataset": "dataset-name",
      "created_at": "timestamp",
      "file_size": "size-string"
    }
  ]
  ```

#### Generate Data
**Endpoint**: `POST /data/generate`  
Starts a data generation job.

- **Request**: None
- **Response**:
  ```json
  {
    "job_id": "generated-job-id"
  }
  ```

### Jobs

#### List Jobs
**Endpoint**: `GET /jobs`  
Fetches the list of all jobs.

- **Response**:
  ```json
  [
    {
      "job_id": "job-id",
      "status": "running/finished/failed",
      "cmd": "command",
      "branch": "branch-name",
      "start_time": "timestamp",
      "end_time": "timestamp"
    }
  ]
  ```

#### Job Status
**Endpoint**: `GET /jobs/{job_id}/status`  
Fetches the status of a specific job.

- **Response**:
  ```json
  {
    "job_id": "job-id",
    "status": "running/finished/failed",
    "branch": "branch-name",
    "command": "command"
  }
  ```

#### Job Logs
**Endpoint**: `GET /jobs/{job_id}/logs`  
Fetches the logs of a specific job.

- **Response**: Text logs of the job.

### Training

#### Start Training
**Endpoint**: `POST /model/train`  
Starts a training job.

- **Request**:
  ```json
  {
    "modelName": "name-of-the-model",
    "branchName": "name-of-the-branch"
  }
  ```
- **Response**:
  ```json
  {
    "job_id": "training-job-id"
  }
  ```

### Pipeline

#### Generate and Train Pipeline
**Endpoint**: `POST /pipeline/generate-train`  
Combines data generation and training into a single pipeline job.

- **Request**:
  ```json
  {
    "modelName": "name-of-the-model",
    "branchName": "name-of-the-branch"
  }
  ```
- **Response**:
  ```json
  {
    "pipeline_job_id": "pipeline-job-id"
  }
  ```

### Model Serving

#### Serve Latest Checkpoint
**Endpoint**: `POST /model/serve-latest`  
Serves the latest model checkpoint on port `8001`.

- **Response**:
  ```json
  {
    "status": "model process started",
    "job_id": "serve-job-id"
  }
  ```

#### Serve Base Model
**Endpoint**: `POST /model/serve-base`  
Serves the base model on port `8000`.

- **Response**:
  ```json
  {
    "status": "model process started",
    "job_id": "serve-job-id"
  }
  ```
```
