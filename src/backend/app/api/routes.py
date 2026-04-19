from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.schemas.common import (
    ModelsResponse,
    PredictionResponse,
    RawApartmentPredictRequest,
    RawFeatureRequest,
    RawHousePredictRequest,
    YesNoType,
)
from app.services.predict import predict_multimodal_from_raw, predict_tabular_from_raw

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request) -> dict[str, bool]:
    registry = request.app.state.registry
    return {"ready": bool(registry.ready)}


@router.get("/v1/models", response_model=ModelsResponse)
def list_models(request: Request) -> ModelsResponse:
    registry = request.app.state.registry
    return ModelsResponse(models=registry.list_models())


@router.post("/v1/predict/tabular/apartment", response_model=PredictionResponse)
def predict_tabular_apartment(payload: RawApartmentPredictRequest, request: Request) -> PredictionResponse:
    return predict_tabular_from_raw(
        property_type="apartment",
        payload=payload,
        request_id=str(uuid4()),
        registry=request.app.state.registry,
    )


@router.post("/v1/predict/tabular/house", response_model=PredictionResponse)
def predict_tabular_house(payload: RawHousePredictRequest, request: Request) -> PredictionResponse:
    return predict_tabular_from_raw(
        property_type="house",
        payload=payload,
        request_id=str(uuid4()),
        registry=request.app.state.registry,
    )


async def _read_images(images: list[UploadFile]) -> list[bytes]:
    image_bytes: list[bytes] = []
    unreadable_count = 0
    empty_count = 0

    for upload in images:
        try:
            payload = await upload.read()
        except Exception:
            unreadable_count += 1
            continue

        if payload:
            image_bytes.append(payload)
        else:
            empty_count += 1

    if not image_bytes and (unreadable_count or empty_count):
        raise HTTPException(
            status_code=422,
            detail=(
                "Uploaded image file(s) were empty or unreadable. "
                "Please upload valid JPG/PNG files using the 'images' field."
            ),
        )

    return image_bytes


def _raw_features_from_form(
    price: float = Form(..., description="Listing price (accepted but ignored during prediction)."),
    rooms: int = Form(..., ge=0, description="Number of rooms"),
    area_m2: float = Form(..., gt=0, description="Property area in m2"),
    land_area_sot: float = Form(..., ge=0, description="Land area in sot"),
    floor: int = Form(..., ge=0, description="Floor number"),
    has_document: YesNoType = Form(..., description="Yes or No"),
    address: str = Form(..., min_length=1, description="Address in 'City, Region' format"),
    temirli: YesNoType = Form(..., description="Yes or No"),
    qaz: YesNoType = Form(..., description="Yes or No"),
    su: YesNoType = Form(..., description="Yes or No"),
    isiq: YesNoType = Form(..., description="Yes or No"),
    avtodayanacaq: YesNoType = Form(..., description="Yes or No"),
    telefon: YesNoType = Form(..., description="Yes or No"),
    internet: YesNoType = Form(..., description="Yes or No"),
    pvc_pencere: YesNoType = Form(..., description="Yes or No"),
    balkon: YesNoType = Form(..., description="Yes or No"),
    kabel_tv: YesNoType = Form(..., description="Yes or No"),
    lift: YesNoType = Form(..., description="Yes or No"),
    kombi: YesNoType = Form(..., description="Yes or No"),
    metbex_mebeli: YesNoType = Form(..., description="Yes or No"),
    merkezi_qizdirici_sistem: YesNoType = Form(..., description="Yes or No"),
    kondisioner: YesNoType = Form(..., description="Yes or No"),
    esyali: YesNoType = Form(..., description="Yes or No"),
    hovuz: YesNoType = Form(..., description="Yes or No"),
    duzelme: YesNoType = Form(..., description="Yes or No"),
) -> RawFeatureRequest:
    return RawFeatureRequest(
        price=price,
        rooms=rooms,
        area_m2=area_m2,
        land_area_sot=land_area_sot,
        floor=floor,
        has_document=has_document,
        address=address,
        temirli=temirli,
        qaz=qaz,
        su=su,
        isiq=isiq,
        avtodayanacaq=avtodayanacaq,
        telefon=telefon,
        internet=internet,
        pvc_pencere=pvc_pencere,
        balkon=balkon,
        kabel_tv=kabel_tv,
        lift=lift,
        kombi=kombi,
        metbex_mebeli=metbex_mebeli,
        merkezi_qizdirici_sistem=merkezi_qizdirici_sistem,
        kondisioner=kondisioner,
        esyali=esyali,
        hovuz=hovuz,
        duzelme=duzelme,
    )


@router.post("/v1/predict/multimodal/apartment", response_model=PredictionResponse)
async def predict_multimodal_apartment(
    request: Request,
    raw_payload: Annotated[RawFeatureRequest, Depends(_raw_features_from_form)],
    images: list[UploadFile] = File(
        ...,
        description=(
            "Property photos. Recommended categories: exterior, living room, kitchen, "
            "bedroom, bathroom, balcony/yard, building entrance, parking/pool."
        ),
    ),
) -> PredictionResponse:
    image_bytes = await _read_images(images)
    if not image_bytes:
        raise HTTPException(status_code=422, detail="At least one image is required for multimodal prediction")

    apartment_payload = RawApartmentPredictRequest(**raw_payload.model_dump())

    return predict_multimodal_from_raw(
        property_type="apartment",
        payload=apartment_payload,
        image_bytes=image_bytes,
        max_images=request.app.state.settings.max_images_per_request,
        request_id=str(uuid4()),
        registry=request.app.state.registry,
    )


@router.post("/v1/predict/multimodal/house", response_model=PredictionResponse)
async def predict_multimodal_house(
    request: Request,
    raw_payload: Annotated[RawFeatureRequest, Depends(_raw_features_from_form)],
    images: list[UploadFile] = File(
        ...,
        description=(
            "Property photos. Recommended categories: exterior, living room, kitchen, "
            "bedroom, bathroom, balcony/yard, building entrance, parking/pool."
        ),
    ),
) -> PredictionResponse:
    image_bytes = await _read_images(images)
    if not image_bytes:
        raise HTTPException(status_code=422, detail="At least one image is required for multimodal prediction")

    house_payload = RawHousePredictRequest(**raw_payload.model_dump())

    return predict_multimodal_from_raw(
        property_type="house",
        payload=house_payload,
        image_bytes=image_bytes,
        max_images=request.app.state.settings.max_images_per_request,
        request_id=str(uuid4()),
        registry=request.app.state.registry,
    )
