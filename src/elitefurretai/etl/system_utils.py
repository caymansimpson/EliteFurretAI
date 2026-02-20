import platform
import warnings

import torch


def is_windows_or_wsl() -> bool:
    return (
        platform.system().lower() == "windows"
        or "microsoft" in platform.uname()[2].lower()
    )


def configure_torch_multiprocessing(
    *,
    use_file_system_sharing: bool = True,
    filter_socket_send_warning: bool = False,
) -> None:
    if use_file_system_sharing and is_windows_or_wsl():
        torch.multiprocessing.set_sharing_strategy("file_system")
    if filter_socket_send_warning:
        warnings.filterwarnings("ignore", message=".*socket.send.*")


def suppress_third_party_warnings(
    *,
    suppress_pydantic_field_warnings: bool = True,
    suppress_socket_send_warning: bool = False,
) -> None:
    if suppress_pydantic_field_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r".*UnsupportedFieldAttributeWarning.*",
            module=r"pydantic\._internal\._generate_schema",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*The 'repr' attribute with value False was provided to the `Field\(\)` function.*",
            module=r"pydantic\._internal\._generate_schema",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*The 'frozen' attribute with value True was provided to the `Field\(\)` function.*",
            module=r"pydantic\._internal\._generate_schema",
        )

    if suppress_socket_send_warning:
        warnings.filterwarnings("ignore", message=".*socket.send.*")
