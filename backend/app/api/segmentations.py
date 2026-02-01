from fastapi import APIRouter, UploadFile, File
from backend.app.services import segmentation_service
from pydantic import BaseModel

router = APIRouter(
    prefix='/segmentations',
    tags=['Segmentations']
)

class SegmentationResponse(BaseModel):
    mask: str
    overlay: str


@router.post('', response_model=SegmentationResponse)
async def create_segmentation(file: UploadFile = File(...)):
    img_bytes = await file.read()
    return segmentation_service.segment_image(img_bytes)