from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Web"])

templates = Jinja2Templates(directory="app/views/templates")


@router.get("/")
async def dashboard(request: Request):
    """Render the monitoring dashboard — the only web page."""
    settings = request.app.state.settings
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "app_name": settings.APP_NAME,
            "version": settings.VERSION,
        },
    )
