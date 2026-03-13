"""
Company Data Automation API Routes
REST API for company data processing and automation
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from ..integrations.company_data_automation import get_automation_service
from ..integrations.lora_adapter_service import get_lora_service
from ..logging_config import get_logger

logger = get_logger("cyrex.api.company_automation")

router = APIRouter(prefix="/company-automation", tags=["company-automation"])


class CompanyDataRequest(BaseModel):
    """Request for company data processing"""
    company_id: str = Field(..., description="Company identifier")
    data: Dict[str, Any] = Field(..., description="Company data to process")
    task_type: str = Field(default="automation", description="Type of automation task")
    use_adapter: bool = Field(default=True, description="Use company-specific LoRA adapter")


class AdapterTrainingRequest(BaseModel):
    """Request for LoRA adapter training"""
    company_id: str = Field(..., description="Company identifier")
    training_data: List[Dict[str, Any]] = Field(..., description="Training data")
    config: Optional[Dict[str, Any]] = Field(None, description="LoRA configuration")
    use_qlora: bool = Field(default=True, description="Use QLoRA (quantized)")


class ToolRegistrationRequest(BaseModel):
    """Request for registering company tools"""
    company_id: str = Field(..., description="Company identifier")
    tools: List[Dict[str, Any]] = Field(..., description="Tool definitions")


@router.post("/process")
async def process_company_data(
    request: CompanyDataRequest,
    automation_service = Depends(get_automation_service),
):
    """Process company data and automate tools"""
    try:
        result = await automation_service.process_company_data(
            company_id=request.company_id,
            data=request.data,
            task_type=request.task_type,
            use_adapter=request.use_adapter,
        )
        return result
    except Exception as e:
        logger.error(f"Company data processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-adapter")
async def train_adapter(
    request: AdapterTrainingRequest,
    lora_service = Depends(get_lora_service),
):
    """Request LoRA adapter training for company"""
    try:
        request_id = await lora_service.request_adapter_training(
            company_id=request.company_id,
            training_data=request.training_data,
            config=request.config,
            use_qlora=request.use_qlora,
        )
        return {
            "request_id": request_id,
            "status": "requested",
            "message": "Adapter training requested, will be processed by Helox",
        }
    except Exception as e:
        logger.error(f"Adapter training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adapters")
async def list_adapters(
    company_id: Optional[str] = None,
    status: Optional[str] = None,
    lora_service = Depends(get_lora_service),
):
    """List LoRA adapters"""
    try:
        adapters = await lora_service.list_adapters(
            company_id=company_id,
            status=status,
        )
        return {"adapters": adapters}
    except Exception as e:
        logger.error(f"Failed to list adapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register-tools")
async def register_tools(
    request: ToolRegistrationRequest,
    automation_service = Depends(get_automation_service),
):
    """Register company-specific tools"""
    try:
        await automation_service.register_company_tools(
            company_id=request.company_id,
            tools=request.tools,
        )
        return {
            "status": "success",
            "message": f"Registered {len(request.tools)} tools for company {request.company_id}",
        }
    except Exception as e:
        logger.error(f"Tool registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

