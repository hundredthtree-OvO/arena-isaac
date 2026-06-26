import functools
import traceback


def safe(fun):
    """Service callback guard that returns response.ret=False on failure."""

    @functools.wraps(fun)
    def wrapper(request, response):
        try:
            return fun(request, response)
        except Exception as exc:  # noqa: BLE001
            try:
                import carb

                carb.log_error(f"Error in {fun.__name__}: {exc}")
            except Exception:
                print(f"Error in {fun.__name__}: {exc}")
            traceback.print_exc()
            if hasattr(response, "ret"):
                response.ret = False
            return response

    return wrapper
