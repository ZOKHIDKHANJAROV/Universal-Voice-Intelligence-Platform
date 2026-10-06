from fastapi import APIRouter

from app.intent.models import IntentResolveRequest, IntentResult
from app.intent.service import get_intent_service

router = APIRouter(prefix="/api/v1/intent", tags=["intent"])


@router.post("/resolve", response_model=IntentResult)
def resolve_intent(request: IntentResolveRequest) -> IntentResult:
    return get_intent_service().resolve(request.text, request.language)
