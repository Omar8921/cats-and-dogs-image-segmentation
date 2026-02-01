from fastapi import APIRouter
from deep_learning import inference

router = APIRouter(
    prefix='/health',
    tags=['Health']
)

@router.get('/')
def health_check():
    return {'status': 'ok', 'model_ready': inference.model_ready}