from custom.mha.mha import MHA
import torch
from typing import Any, Callable, Iterable
import torch._dynamo as dynamo
from torch import _guards

def custom_backend(
    graph_module: torch.fx.GraphModule,
    example_inputs: list[Any]
) -> Callable[..., Any]:
    print("new variant!")
    print(graph_module.code)
    return graph_module.forward

def guard_export_fn(
    # torch._guards
    guards: _guards.GuardsSet
) -> None:
    print("==GUARDS==")
    print(len(guards))
    # for guard in guards:
    #     print(guard)

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 16
    S = 1024
    D = 128
    H = 128
    N = 4
    
    net = MHA(D, H, N).eval()
    # compiled_net = torch.compile(net, backend=custom_backend)
    # torch._dynamo.eval_frame.py
    compiled_net = dynamo.optimize(
        backend=custom_backend,
        guard_export_fn=guard_export_fn
    )(net)

    for idx in range(10):
        x = torch.randn(size=(B * (2 if idx % 2 == 0 else 1), S, D))
        with torch.inference_mode():
            out = compiled_net(x)
    #print(out)
    # print(out.shape)
