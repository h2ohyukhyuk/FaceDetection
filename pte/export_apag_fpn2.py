"""Export/reuse two fixed-shape FP32 XNNPACK PTEs and benchmark CPU inference.

Requires a compatible torch + executorch environment (Python >=3.10).
Default checkpoint matches infer/infer_apag_fpn2.py. Shapes below are H x W.
"""
import argparse
import json
import os
import shutil
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHAPES = ((256, 384), (192, 288))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/home/j/models/face_detection/apag-fpn2-0000/180.pt'))
    parser.add_argument('--output-dir', type=Path, default=ROOT/'onnx__')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--threads', type=int, default=1, help='XNNPACK CPU thread count')
    parser.add_argument('--seed', type=int, default=1024)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1 or args.threads < 1:
        parser.error('warmup >= 0, runs >= 1 and threads >= 1 are required')
    return args


def load_model(checkpoint, torch):
    from net.apag_fpn_net2 import Network
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        state = state.get('model', state.get('state_dict', state))
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model = Network().eval()
    # A missing/incorrect weight must not silently export a partially trained model.
    model.load_state_dict(state, strict=True)
    return model


def export_if_missing(model, sample, path, torch):
    if path.exists():
        print(f'[reuse] {path}', flush=True)
        return False
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    # Absolute interpreter launches do not add the venv/bin directory to PATH.
    # Older ExecuTorch wheels invoke flatc by name instead of locating the bundle.
    if 'FLATC_EXECUTABLE' not in os.environ:
        flatc = shutil.which('flatc')
        if flatc is None:
            sibling = Path(sys.executable).parent/'flatc'
            if sibling.is_file():
                flatc = str(sibling)
        if flatc:
            os.environ['FLATC_EXECUTABLE'] = flatc
    print(f'[export] {path}', flush=True)
    exported = torch.export.export(model, (sample,))
    program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]).to_executorch()
    # Exclusive creation also prevents overwriting an artifact created concurrently.
    try:
        with path.open('xb') as output:
            try:
                program.write_to_file(output)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
    except FileExistsError:
        print(f'[reuse] Another process already created {path}', flush=True)
        return False
    return True


def benchmark(method, inputs, warmup, runs):
    for i in range(warmup):
        method.execute([inputs[i % len(inputs)]])
    timings = []
    for i in range(runs):
        sample = inputs[i % len(inputs)]
        start = time.perf_counter_ns()
        output = method.execute([sample])
        timings.append((time.perf_counter_ns()-start)/1e6)
    return timings, output


def main():
    args = parse_args()
    try:
        import torch
        from executorch.runtime import Runtime
        from executorch.extension.pybindings import portable_lib
    except ImportError as error:
        raise SystemExit('Use a compatible torch + executorch environment. '
                         'Tested: /home/j/projects/executorch/venv/bin/python '
                         '(torch 2.12.1, executorch 1.3.1).\n'+str(error)) from error
    # ExecuTorch uses its own pool: torch.set_num_threads alone is insufficient.
    # Configure it once before loading any runtime methods or creating workers.
    if not hasattr(portable_lib, '_unsafe_reset_threadpool'):
        raise RuntimeError('This ExecuTorch build cannot configure the runtime thread pool')
    portable_lib._unsafe_reset_threadpool(args.threads)
    actual_threads = portable_lib._threadpool_get_thread_count()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, torch)
    runtime = Runtime.get()
    report = {
        'checkpoint': str(args.checkpoint.resolve()), 'backend': 'XNNPACK CPU FP32',
        'torch': torch.__version__, 'executorch': version('executorch'),
        'machine': platform.machine(), 'platform': platform.platform(),
        'threads': actual_threads, 'seed': args.seed, 'warmup': args.warmup,
        'runs': args.runs, 'timing': 'Python method.execute only; excludes load/export/input generation/preprocessing/postprocessing',
        'results': [],
    }
    print(f"XNNPACK CPU FP32 | torch={torch.__version__} executorch={report['executorch']} "
          f'threads={actual_threads} warmup={args.warmup} runs={args.runs}', flush=True)
    with torch.inference_mode():
        for height, width in SHAPES:
            # Cycle through pre-generated random inputs without timing RNG/allocation.
            inputs = [torch.rand(1, 3, height, width)-.5 for _ in range(8)]
            path = args.output_dir/f'apag_fpn2_{args.checkpoint.stem}_{height}x{width}.pte'
            created = export_if_missing(model, inputs[0], path, torch)
            program = runtime.load_program(str(path))
            method = program.load_method('forward')
            max_error = 0.
            for sample in inputs[:2]:
                expected = model(sample)
                actual = method.execute([sample])[0]
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
                max_error = max(max_error, (actual-expected).abs().max().item())
            timings, output = benchmark(method, inputs, args.warmup, args.runs)
            values = torch.tensor(timings, dtype=torch.float64)
            result = {
                'input_shape': list(inputs[0].shape), 'output_shape': list(output[0].shape),
                'pte': str(path.resolve()), 'exported': created, 'bytes': path.stat().st_size,
                'max_abs_error': max_error, 'mean_ms': values.mean().item(),
                'median_ms': values.median().item(), 'p90_ms': torch.quantile(values, .9).item(),
                'min_ms': values.min().item(), 'max_ms': values.max().item(),
            }
            report['results'].append(result)
            print(f"{height}x{width}: mean={result['mean_ms']:.3f} ms, "
                  f"median={result['median_ms']:.3f} ms, p90={result['p90_ms']:.3f} ms, "
                  f"max_abs_error={max_error:.6g}", flush=True)
            del method, program
    report_path = args.output_dir/f'apag_fpn2_{args.checkpoint.stem}_benchmark.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
