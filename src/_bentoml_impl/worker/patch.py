import importlib.abc
import importlib.machinery
import logging
import os
import subprocess
import sys
import types

logger = logging.getLogger("bentoml.worker.service")


# Custom loader that calls the callback
class SafeTensorLoader(importlib.abc.Loader):
    def __init__(self, original_loader: importlib.abc.Loader):
        self.original_loader = original_loader

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return self.original_loader.create_module(spec)

    def exec_module(self, module: types.ModuleType):
        self.original_loader.exec_module(module)
        # Call the callback after the module is executed
        _do_patch(module)


# Custom finder that uses the custom loader
class SafeTensorFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "safetensors.torch":
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    if spec.loader is not None:
                        # Replace the loader with our custom loader
                        spec.loader = SafeTensorLoader(spec.loader)
                    return spec
        return None


def _do_patch(target_module):
    logger.info(
        "Patching safetensors.torch.safe_open to preheat model loading in parallel"
    )

    # Save the original safe_open class
    OriginalSafeOpen = target_module.safe_open

    # Define a new class to wrap around the original safe_open class
    class PatchedSafeOpen:
        def __init__(self, filename, framework, device="cpu"):
            # Call the read_ahead method before the usual safe_open
            self.read_ahead(filename)

            # Initialize the original safe_open
            self._original_safe_open = OriginalSafeOpen(filename, framework, device)

        def __enter__(self):
            return self._original_safe_open.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            return self._original_safe_open.__exit__(exc_type, exc_value, traceback)

        @staticmethod
        def read_ahead(
            file_path,
            num_processes=None,
            size_threshold=100 * 1024 * 1024,
            block_size=1024 * 1024,
        ):
            """
            Read a file in parallel using multiple processes.

            Args:
                file_path: Path to the file to read
                num_processes: Number of processes to use for reading the file. If None, the number of processes is set to the number of CPUs.
                size_threshold: If the file size is smaller than this threshold, only one process is used to read the file.
                block_size: Block size to use for reading the file
            """
            if num_processes is None:
                num_processes = os.cpu_count() or 8

            file_size = os.path.getsize(file_path)
            if file_size <= size_threshold:
                num_processes = 1

            chunk_size = file_size // num_processes
            processes = []

            for i in range(1, num_processes):
                start_byte = i * chunk_size
                end_byte = (
                    start_byte + chunk_size if i < num_processes - 1 else file_size
                )
                logger.info(
                    f"Reading bytes {start_byte} to {end_byte} from {file_path}"
                )
                process = subprocess.Popen(
                    [
                        "dd",
                        f"if={file_path}",
                        "of=/dev/null",
                        f"bs={block_size}",
                        f"skip={start_byte // block_size}",
                        f"count={(end_byte - start_byte) // block_size}",
                        "status=none",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                processes.append(process)

        def __getattr__(self, name):
            return getattr(self._original_safe_open, name)

    # Patch the original safetensors.torch module directly
    target_module.safe_open = PatchedSafeOpen


# Insert the custom finder at the beginning of sys.meta_path
def patch_safetensor():
    sys.meta_path.insert(0, SafeTensorFinder())
