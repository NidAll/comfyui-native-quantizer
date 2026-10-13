"""Process exit codes and cancellation for CLI invocation."""
import os
import sys

from native_quantizer import events
from native_quantizer.errors import QuantizerError, UsageError


def entrypoint(argv=None) -> int:
    try:
        from native_quantizer.cli.commands import main
        code = main(argv)
        events.emit('completion', ok=code == 0, exit_code=code)
        return code
    except BrokenPipeError:
        # Avoid another exception when Python flushes stdout after a pipe closes.
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, sys.stdout.fileno())
        os.close(fd)
        return 141
    except UsageError as exc:
        events.emit('completion', ok=False, exit_code=2, error=str(exc))
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    except (QuantizerError, OSError, ValueError) as exc:
        events.emit('completion', ok=False, exit_code=1, error=str(exc))
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        if exc.name not in {'torch', 'numpy', 'safetensors'}:
            raise
        print('comfyui_native_quantizer requires torch (>=2.1), safetensors and numpy. '
              f'Missing dependency: {exc.name}', file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        events.emit('completion', ok=False, exit_code=130, error='cancelled')
        print('Cancelled.', file=sys.stderr)
        return 130
