from typing import List

from fastapi import APIRouter

import yatai.schema.bundle as bundle_schema

router = APIRouter()


@router.get("/{name}/version/{version}", response_model=bundle_schema.GetBundleResponse)
def get_bundle(name: str, version: str):
    return {}


@router.delete("/{name}/version/{version}")
def delete_bundle(name: str, version: str):
    return {}


@router.get("/{name}")
def list_bundle_versions(name: str):
    return []


@router.get("/", response_model=bundle_schema.ListBundleResponse)
def list_bundles(
    label_selectors: dict = None,
    skip: int = 0,
    limit: int = 20,
    order_by: str = None,
    ascending_order: bool = False,
):
    return []


@router.post(
    "/", response_model=bundle_schema.AddBundleResponse
)  # or /{name}/version/{version}
def add_bundle():
    return {}
