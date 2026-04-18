import argparse
import time
import torch

def compute_normal_eq(X, y):
    return torch.linalg.inv(X.T @ X) @ X.T @ y

def compute_lstsq(X, y):
    return torch.linalg.lstsq(X, y).solution

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\nRunning on: {device}")
    print(f"n={args.n}, d={args.d}, method={args.method}, iters={args.iters}")

    # Create data
    X = torch.rand(args.n, args.d, device=device)
    true_theta = torch.rand(args.d, 1, device=device)
    y = X @ true_theta + 0.01 * torch.rand(args.n, 1, device=device)

    # Warmup (important for GPU)
    for _ in range(args.warmup):
        if args.method == "lstsq":
            _ = compute_lstsq(X, y)
        else:
            _ = compute_normal_eq(X, y)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run
    start = time.time()

    for _ in range(args.iters):
        if args.method == "lstsq":
            theta = compute_lstsq(X, y)
        else:
            theta = compute_normal_eq(X, y)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    print(f"Total Time: {end - start:.4f} sec")
    print(f"Avg Time per iter: {(end - start)/args.iters:.6f} sec")

    # Optional correctness check
    if args.check:
        theta_lstsq = compute_lstsq(X, y)
        theta_normal = compute_normal_eq(X, y)
        print("Close:", torch.allclose(theta_lstsq, theta_normal, atol=1e-3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=100000, help="number of samples")
    parser.add_argument("--d", type=int, default=128, help="number of features")
    parser.add_argument("--iters", type=int, default=50, help="profiling iterations")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument("--method", type=str, choices=["lstsq", "normal"], default="lstsq")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--check", action="store_true", help="compare both methods")

    args = parser.parse_args()
    main(args)