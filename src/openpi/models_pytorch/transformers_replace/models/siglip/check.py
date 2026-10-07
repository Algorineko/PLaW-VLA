import packaging.version
import transformers


def check_whether_transformers_replace_is_installed_correctly() -> bool:
    """True only when the PLaW-VLA patched files are actually installed.

    A version check alone false-positives on a stock transformers 5.x install.
    The patched paligemma module additionally defines
    `PaligemmaModelOutputWithPast`, which does not exist upstream, so its
    presence proves the patched files were copied over.
    """
    try:
        if packaging.version.parse(transformers.__version__).major != 5:
            return False
        from transformers.models.paligemma import modeling_paligemma

        return hasattr(modeling_paligemma, "PaligemmaModelOutputWithPast")
    except (packaging.version.InvalidVersion, ImportError):
        return False
