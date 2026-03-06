def ensure_batch_agent(obs: torch.Tensor) -> torch.Tensor:
    """
    Make obs shape always (B, N, ...)
    - If obs is (N, ...): treat as (B=1, N, ...)
    - If obs is (B, N, ...): keep
    - If obs is (B, ...): treat as (B, N=1, ...)
    """
    if obs.dim() >= 2 and obs.shape[0] == 1 and obs.dim() == 2:
        # ambiguous, but keep as (B=1, D) -> (1,1,D)
        return obs.unsqueeze(1)
    if obs.dim() >= 2 and obs.dim() == 2:
        # (N, D) -> (1, N, D)
        return obs.unsqueeze(0)
    if obs.dim() >= 3:
        # could be (B,N,...) OR (N,C,H,W) OR (B,C,H,W)
        # Heuristic:
        # - if 2nd dim looks like channels (<=8) and last two dims exist -> assume (B,C,H,W) => add N=1
        if obs.dim() == 4 and obs.shape[1] <= 8:
            # (B,C,H,W) -> (B,1,C,H,W)
            return obs.unsqueeze(1)
        # if obs.dim()==4 and obs.shape[0] <= 8 and obs.shape[1] > 8, could be (N,D1,D2,D3) uncommon.
        # assume already (B,N,...) for dim>=3 except the (B,C,H,W) case above
        return obs
    # (D,) -> (1,1,D)
    return obs.unsqueeze(0).unsqueeze(0)


def flatten_BN(x: torch.Tensor) -> torch.Tensor:
    """
    (B,N,...) -> (B*N,...)
    """
    B, N = x.shape[0], x.shape[1]
    return x.reshape(B * N, *x.shape[2:])


def infer_obs_and_action(env):
    # reset compat: some envs return (obs, info), some return obs only
    out = env.reset()
    obs0 = out[0] if isinstance(out, (tuple, list)) else out

    # obs0 expected to be (N, ...) for MAPPO
    assert isinstance(obs0, np.ndarray), f"obs0 must be np.ndarray, got {type(obs0)}"
    assert obs0.ndim in (2, 4), f"Expected obs shape (N,D) or (N,C,H,W), got {obs0.shape}"

    n_agents = obs0.shape[0]
    obs_shape = tuple(obs0.shape[1:])  # (D,) or (C,H,W)

    # infer act_dim
    act_dim = None
    # PettingZoo style
    if hasattr(env, "agents") and hasattr(getattr(env, "env", None), "action_space"):
        try:
            act_dim = env.env.action_space(env.agents[0]).n
        except Exception:
            pass

    if act_dim is None:
        # Gym-style
        aspace = getattr(getattr(env, "env", env), "action_space", None)
        if aspace is None:
            raise RuntimeError("Cannot infer action_space from env.")
        if hasattr(aspace, "spaces"):
            act_dim = aspace.spaces[0].n
        else:
            act_dim = aspace.n

    return obs_shape, n_agents, act_dim