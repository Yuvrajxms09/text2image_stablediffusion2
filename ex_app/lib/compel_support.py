import logging

try:
    from compel import CompelForSDXL
except ImportError:  # optional dependency
    CompelForSDXL = None


logger = logging.getLogger(__name__)


def init_sdxl_compel(pipe, device: str):
    if CompelForSDXL is None:
        logger.info("compel is not available; using raw prompt fallback")
        return None

    logger.info("compel is active for long prompt conditioning")
    return CompelForSDXL(pipe, device=device)


def build_prompt_conditioning(prompt: str, compel):
    if compel is None:
        return None

    return compel(prompt)
