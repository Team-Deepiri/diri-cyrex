# LangGraph Workflow Playground - Testing Interface

## Overview

The **WorkflowPlayground** component provides a comprehensive testing interface for the LangGraph multi-agent workflow system. It allows you to test different workflow types, visualize execution in real-time, and monitor agent performance.

## Features

### 1. **Workflow Configuration**
- Select workflow type:
  - **Standard**: Task → Plan → Code → Quality
  - **Lease Abstraction**: Process lease documents
  - **Contract Intelligence**: Analyze contracts
  - **Vendor Fraud Detection**: Detect fraud
  - **Custom**: Manual configuration
- Configure task description
- Set context (document text, URLs, IDs)
- Optional session and user IDs

### 2. **Real-Time Execution**
- Stream workflow execution
- View agent transitions in real-time
- See tool calls and results
- Monitor errors and warnings

### 3. **Visual Workflow Graph**
- Visual representation of agent nodes
- Status indicators (pending, active, completed, error, skipped)
- Click nodes to view details
- See agent history and responses

### 4. **Results Display**
- View final workflow results
- See generated plans, code, and quality checks
- Export workflow state as JSON
- Review agent history

### 5. **Monitoring Dashboard**
- Metrics: workflow type, agents used, tool calls, errors
- Agent history with timestamps
- Full workflow state JSON
- Error log

## Usage

### Accessing the Playground

1. Open the cyrex-interface
2. Click **"LangGraph Workflow"** in the sidebar
3. The WorkflowPlayground will open

### Testing a Standard Workflow

1. **Configure Tab**:
   - Select "Standard Workflow"
   - Enter task: "Create a Python function to calculate fibonacci numbers"
   - Click "Execute Workflow"

2. **Execute Tab**:
   - Watch real-time execution messages
   - See workflow progress

3. **Visualize Tab**:
   - View the workflow graph
   - Click nodes to see details
   - Review results (plan, code, quality check)

4. **Monitor Tab**:
   - Check metrics
   - Review agent history
   - View full state

### Testing a Lease Abstraction Workflow

1. **Configure Tab**:
   - Select "Lease Abstraction"
   - Enter task description
   - Add context JSON:
     ```json
     {
       "document_text": "...",
       "document_url": "s3://bucket/lease.pdf",
       "lease_id": "lease_123"
     }
     ```
   - Click "Execute Workflow"

2. **Visualize Tab**:
   - See lease processor node
   - View abstraction results
   - Check confidence scores

### Testing a Contract Intelligence Workflow

1. **Configure Tab**:
   - Select "Contract Intelligence"
   - Enter task description
   - Add context JSON:
     ```json
     {
       "document_text": "...",
       "document_url": "s3://bucket/contract.pdf",
       "contract_id": "contract_123",
       "contract_number": "CONTRACT-2024-001",
       "party_a": "Acme Corp",
       "party_b": "Tech Services Inc"
     }
     ```
   - Click "Execute Workflow"

2. **Visualize Tab**:
   - See contract processor node
   - View extracted clauses
   - Check obligation dependencies

### Testing a Fraud Detection Workflow

1. **Configure Tab**:
   - Select "Vendor Fraud Detection"
   - Enter task: "Analyze this invoice for potential fraud"
   - Add context with invoice data
   - Click "Execute Workflow"

2. **Visualize Tab**:
   - See fraud agent node
   - View fraud analysis results
   - Check risk scores

## API Endpoints

### Execute Workflow
```
POST /api/workflow/execute
Content-Type: application/json

{
  "task_description": "Process this lease document",
  "workflow_type": "lease",
  "context": {
    "document_text": "...",
    "document_url": "...",
    "lease_id": "..."
  },
  "session_id": "session_123",
  "user_id": "user_123",
  "stream": true
}
```

### Get Workflow Types
```
GET /api/workflow/types

Response:
[
  {
    "id": "standard",
    "name": "Standard Workflow",
    "description": "Task → Plan → Code → Quality",
    "agents": ["task_agent", "plan_agent", "code_agent", "qa_agent"]
  },
  ...
]
```

### Workflow Health
```
GET /api/workflow/health

Response:
{
  "status": "healthy",
  "workflow_available": true,
  "langgraph_available": true,
  "checkpointing_available": true
}
```

## Workflow Types

### Standard Workflow
- **Agents**: Task Decomposer → Time Optimizer → Creative Sparker → Quality Assurance
- **Use Case**: General task automation, code generation
- **Output**: Plan, code, quality check

### Lease Abstraction
- **Agents**: Lease Processor
- **Use Case**: Extract structured data from lease documents
- **Output**: Abstracted terms, obligations, financial terms

### Contract Intelligence
- **Agents**: Contract Processor
- **Use Case**: Analyze contracts, track clause evolution
- **Output**: Clauses, obligations, dependency graphs

### Vendor Fraud Detection
- **Agents**: Vendor Intelligence Agent
- **Use Case**: Detect fraud in invoices and vendor relationships
- **Output**: Fraud analysis, risk scores, recommendations

## Visual Indicators

### Node Status
- **Pending** ⏳: Not yet executed
- **Active** 🔄: Currently executing (pulsing animation)
- **Completed** ✅: Successfully completed
- **Error** ❌: Execution failed
- **Skipped** ⏭️: Skipped due to workflow routing

### Workflow Status
- **Executing**: Workflow is running
- **Completed**: Workflow finished successfully
- **Error**: Workflow encountered errors

## Tips

1. **Start Simple**: Test with standard workflow first
2. **Check Context**: Ensure context JSON is valid for specialized workflows
3. **Monitor Errors**: Check the Monitor tab for detailed error information
4. **Export State**: Use "Export State" to save workflow results
5. **Review History**: Check agent history to understand execution flow

## Troubleshooting

### Workflow Not Executing
- Check that Ollama is running
- Verify API base URL is correct
- Check browser console for errors

### No Results Displayed
- Ensure workflow completed successfully
- Check for errors in the Monitor tab
- Verify context data is correct

### Streaming Not Working
- Check network connection
- Verify API supports streaming
- Check browser console for errors

## Files

- **Component**: `cyrex-interface/src/components/WorkflowPlayground/WorkflowPlayground.tsx`
- **Styles**: `cyrex-interface/src/components/WorkflowPlayground/WorkflowPlayground.css`
- **API Routes**: `app/routes/workflow_api.py`
- **Backend**: `app/core/langgraph_workflow.py`

## Integration

The WorkflowPlayground is integrated with:
- ✅ LangGraph workflow system
- ✅ All agent types (task, plan, code, QA, lease, contract, fraud)
- ✅ Memory manager
- ✅ Session manager
- ✅ Event registry
- ✅ RAG pipeline
- ✅ Tool registry

