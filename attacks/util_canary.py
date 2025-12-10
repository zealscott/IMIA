import numpy as np
from utils.loader import load_model
import torch
from attacks.util_learn_canary import *


def convert_to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    else:
        return x.unsqueeze(0)


def learning_canary(
    lira_scores, args, shadow_model_info_list, target_model_path, train_data, device="cuda"
):
    n_samples = len(train_data)
    shadow_stats, shadow_indices = collect_all_logits(shadow_model_info_list, n_samples)
    mu_in, sigma_in, mu_out, sigma_out = shadow_stats
    in_model_indices, out_model_indices = shadow_indices

    # load target model
    target_model = load_model(args)
    target_model.load_state_dict(torch.load(target_model_path, map_location=device))
    target_model.to(device)
    target_model.eval()

    shadow_models = load_all_shadow_models(args, shadow_model_info_list, device)
    final_scores = {}

    for i in range(128):
        i_in = in_model_indices[i]
        i_out = out_model_indices[i]
        in_shadow_models = [shadow_models[idx] for idx in i_in if idx < len(shadow_models)]
        out_shadow_models = [shadow_models[idx] for idx in i_out if idx < len(shadow_models)]

        canary_scores = []
        canary_logits = []
        # get the lira score for the original image
        for _ in range(args.ensemble_k):
            x, y = train_data[i]
            x_t = convert_to_tensor(x).to(device)
            # get the lira score
            target_logit = cal_logit(target_model, x_t, y)
            # query the shadow models
            shadow_in_logits = []
            shadow_out_logits = []
            for model in in_shadow_models:
                shadow_in_logits.append(cal_logit(model, x_t, y))
            for model in out_shadow_models:
                shadow_out_logits.append(cal_logit(model, x_t, y))

            mu_in = np.median(shadow_in_logits)
            sigma_in = np.std(shadow_in_logits)
            mu_out = np.median(shadow_out_logits)
            sigma_out = np.std(shadow_out_logits)
            mem_score = cal_lira_score(target_logit, mu_in, sigma_in, mu_out, sigma_out)
            # canary_scores.append(mem_score)

        # # get the lira score for the canary image
        x_np, y = train_data[i]
        x_t = convert_to_tensor(x_np).to(device)
        x_init = x_t.clone()
        y_t = torch.tensor([y], dtype=torch.int64, device=device)
        x_canary = canary_optimize(
            x_init,
            y_t,
            in_shadow_models,
            out_shadow_models,
            args,
            device=device,
            steps=args.iter,
            lr=args.lr,
            eps=args.eps,
        )
        # Evaluate
        logit_canary = cal_logit(target_model, x_canary, y)
        canary_logits.append(logit_canary)
        mem_score = cal_lira_score(logit_canary, mu_in, sigma_in, mu_out, sigma_out)
        canary_scores.append(mem_score)

        # average log-odds
        lira_score = lira_scores[i]
        canary_score = np.mean(canary_scores)
        print(f"finished {i} samples")
        print(f"canary_score: {canary_score}, lira_score: {lira_score}")
        final_scores[i] = np.mean([canary_score, lira_score])
        # final_scores[i] = lira_prob
    return final_scores
