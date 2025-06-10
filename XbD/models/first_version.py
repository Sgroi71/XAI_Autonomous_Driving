import torch
import torch.nn as nn

class XbD_FirstVersion(nn.Module):
    def __init__(self, num_classes: int, N: int):
        """
        Args:
            d_model: dimensionality to project each 41‐dim vector into
            N: number of objects per time step (so that lin2’s input size = N*d_model)
        """
        super().__init__()
        self.N = N
        self.num_classes = num_classes

        # Second linear: (N * d_model) → 7
        self.lin2 = nn.Linear(N * self.num_classes, 7)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Tensor of shape (batch_size, T, N, num_classes)

        Returns:
            logits: Tensor of shape (batch_size, T, 7)
        """
        batch_size, T, N, _ = labels.shape
        assert N == self.N, f"Expected N={self.N}, but got N={N}"

        # Flatten (B, T, N, num_classes) → (B, T, N*num_classes)
        x = labels.reshape(batch_size, T, N * self.num_classes)

        # 5) Apply lin2 to each time step. First, merge (B, T, N*self.num_classes) → (B*T, N*self.num_classes)
        x = x.view(batch_size * T, N * self.num_classes)  # → (B*T, N*self.num_classes)
        logits = self.lin2(x)                            # → (B*T, 7)

        # 6) Reshape back to (B, T, 7)
        logits = logits.view(batch_size, T, 7)
        return logits

    def explain_contributions(
            model: nn.Module,
            labels: torch.Tensor,
            concept_names,
            ego_action_names,
            top_k_dets: int = 5,
            top_k_actions: int = 1,
            top_k_locations: int = 1
        ):
        A_START, A_END     = 0, 10
        Ac_START, Ac_END   = 10, 29
        L_START, L_END     = 29, 41

        # final‐layer weights: shape (7, N*C)
        W = model.lin2.weight

        with torch.no_grad():
            logits = model(labels)        # (B, T, 7)
        preds = logits.argmax(dim=-1)     # (B, T)

        B, T, N, C = labels.shape
        labels_flat = labels.reshape(B, T, N * C)
        explanations = []

        for b in range(B):
            batch_expls = []
            for t in range(T):
                # 1) which ego‐class?
                pred_class = preds[b, t].item()
                pred_label = ego_action_names[pred_class]

                # 2) get contribution vector for that class
                w_row = W[pred_class]              # (N*C,)
                c_row = labels_flat[b, t]          # (N*C,)
                contrib_flat = w_row * c_row       # (N*C,)
                contrib_mat = contrib_flat.view(N, C)
                conf_mat = c_row.view(N, C)

                # 3) find which detections are “present”
                valid = (labels[b, t].sum(dim=-1) > 0).tolist()
                valid_indices = [i for i, v in enumerate(valid) if v]

                # 4) rank by total absolute contribution
                det_strengths = {
                    i: contrib_mat[i].abs().sum().item()
                    for i in valid_indices
                }
                top_dets = sorted(det_strengths, key=det_strengths.get, reverse=True)[:top_k_dets]

                # 5) for each top det, pick out agent/actions/locations
                det_expls = []
                for det in top_dets:
                    row = contrib_mat[det]
                    conf_row = conf_mat[det]

                    # agent
                    a_vals = row[A_START:A_END]
                    a_local = a_vals.abs().argmax().item()
                    a_idx = A_START + a_local
                    agent_name = concept_names[a_idx]
                    agent_contrib = row[a_idx].item()
                    agent_conf = conf_row[a_idx].item()

                    # actions
                    act_vals = row[Ac_START:Ac_END]
                    top_acts = act_vals.abs().topk(
                        min(top_k_actions, Ac_END-Ac_START)
                    ).indices.tolist()
                    actions = [
                        (
                            concept_names[Ac_START + i],
                            row[Ac_START + i].item(),
                            conf_row[Ac_START + i].item()
                        )
                        for i in top_acts
                    ]

                    # locations
                    loc_vals = row[L_START:L_END]
                    top_locs = loc_vals.abs().topk(
                        min(top_k_locations, L_END-L_START)
                    ).indices.tolist()
                    locations = [
                        (
                            concept_names[L_START + i],
                            row[L_START + i].item(),
                            conf_row[L_START + i].item()
                        )
                        for i in top_locs
                    ]

                    det_expls.append({
                        'detection': det,
                        'agent':     (agent_name, agent_contrib, agent_conf),
                        'actions':   actions,
                        'locations': locations
                    })

                # 6) assemble per‐time‐step
                batch_expls.append({
                    'predicted_ego': pred_label,
                    'detections':   det_expls
                })

            explanations.append(batch_expls)

        return explanations