import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns


def compute_lrp_explanation(model_components, model_inputs, model_case, logit_function=lambda x: x,
                            return_gradient_norm=False, tokenizer=None, generation_ids_dict=None):
    if 'roberta' in model_case:
        embeddings = model_components['embeddings'](**model_inputs)
        embeddings_ = embeddings.detach().requires_grad_(True)
        logits = model_components['classifier'](model_components['encoder'](embeddings_).last_hidden_state)

    elif 'gpt2' in model_case:

        batch_size = model_inputs['input_ids'].shape[0]
        sequence_lengths = -1  # assumes no padding

        token_embeddings = model_components['embeddings'](model_inputs['input_ids'])

        position_ids = torch.tensor([[int(i) for i in range(model_inputs['input_ids'].shape[-1])]]).to(
            model_inputs['input_ids'].device)

        position_embeds = model_components['wpe'](position_ids)

        embeddings = position_embeds + token_embeddings

        embeddings_ = embeddings.detach().requires_grad_(True)

        model_inputs = {'input_ids': None,
                        'past_key_values': None,
                        'use_cache': False,
                        'position_ids': position_ids,
                        'attention_mask': None,
                        'inputs_embeds': embeddings_,
                        'token_type_ids': None}

        logits = model_components['classifier'](model_components['encoder'](**model_inputs).last_hidden_state)
        logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]  # gpt2 pooling

    elif 't5' in model_case:

        model_xai = model_components['model']
        input_ids = model_inputs['input_ids']
        if False:
            extra_input_ids = tokenizer("sentiment: ", return_tensors="pt").input_ids[:, :-1].to(input_ids.device)
            inputs_conditional = torch.concat([extra_input_ids, input_ids], 1)
        else:
            inputs_conditional = input_ids

        model_xai.return_gradient_norm = return_gradient_norm

        preds = model_xai.generate_xai(inputs_conditional).squeeze()  # , explain_ids = [3, 29])

        # print('Generated', preds)
        # print('Generated', ' '.join(tokenizer.convert_ids_to_tokens(preds)))

        generation_ids = list(generation_ids_dict.values())

        outs = [(rel, ls, pred) for rel, ls, pred in zip(model_xai.Rs, model_xai.Ls, model_xai.Gids) if
                pred in generation_ids]

        assert len(outs) == 1

        relevance = outs[0][0]
        selected_logit = outs[0][1]
        predicted = outs[0][2]

        logits = np.array(model_xai.Ls).squeeze()

        # assert return_gradient_norm == False  # doesn't work for norm yet

        return relevance, selected_logit, logits

    # Select what signal to propagate back through the network
    selected_logit = logit_function(logits)
    selected_logit.sum().backward()

    gradient = embeddings_.grad

    logits = logits.squeeze().detach().cpu().numpy()

    selected_logit = selected_logit.detach().cpu().numpy().sum()
    if return_gradient_norm:
        # Compute L1 norm over hidden dim
        gradient = gradient[0, :].squeeze().detach().cpu().numpy()
        relevance = np.linalg.norm(gradient, 1, -1)
        return relevance, selected_logit, logits

    else:
        # Compute lrp explanation  
        relevance = gradient * embeddings_
        relevance = relevance[0, :].sum(1).squeeze().detach().cpu().numpy()
        return relevance, selected_logit, logits


def plot_conservation(Ls, Rs, filename=None):
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(Ls, Rs, s=20, label='R', c='blue')
    ax.plot(Ls, Ls, color='black', linestyle='-', linewidth=1)

    # ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=True)
    # ax.set_xlabel('output $f$', fontsize=30,  usetex=True)
    ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=False)
    ax.set_xlabel('output $f$', fontsize=30, usetex=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    f.tight_layout()
    if filename:
        f.savefig(filename, dpi=100)
        plt.close()
    else:
        plt.show()


def plot_conservation_all(Ls, Rs, Rattns, filename=None):
    from pylab import cm
    cmap = cm.get_cmap('plasma')  # , 5)

    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    for i in range(len(Rattns.keys())):  # [0, 3, 6, 9, 10, 11]:
        ax.scatter(Ls, Rattns[i], s=20, label=str(i), c=cmap(0.1 + i / 12))  # , edgecolor='grey') #, fill)

    ax.scatter(Ls, Rs, s=20, label='R', c='blue')
    ax.plot(Ls, Ls, color='black', linestyle='-', linewidth=1)

    # ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=True)
    # ax.set_xlabel('output $f$', fontsize=30,  usetex=True)
    ax.set_ylabel('$\sum_i R_i$', fontsize=30, usetex=False)
    ax.set_xlabel('output $f$', fontsize=30, usetex=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    f.tight_layout()
    if filename:
        f.savefig(filename, dpi=100)
        plt.close()
    else:
        plt.show()


def get_canvas(words, x, H=200, W=50):
    ntoks = len(''.join(words))
    W_all = W * ntoks
    fracs = [len(w_) / ntoks for w_ in words]
    delta_even = int(W_all / ntoks)

    X = np.zeros((H, W_all))
    x0 = 0

    x_centers = []
    for i, (w_, b) in enumerate(zip(words, x)):
        delta = int((len(w_) / ntoks) * W_all)

        delta = int((0.85 * delta_even + 0.15 * delta))
        X[:, x0:x0 + delta] = b

        x_centers.append(x0 + int(delta / 2))
        x0 = x0 + delta

    X = X[:, :x0]
    return X, x_centers


def plot_sentence(words, x, H0=100, W0=52, fax=None):
    if fax is None:
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2),
                             gridspec_kw={'height_ratios': [1]})
    else:
        f, ax = fax
    x_, x_centers = get_canvas(words, x, H=H0, W=W0)
    h = ax.imshow(x_, cmap='bwr', vmin=-1, vmax=1., alpha=1.)

    for k, word in zip(x_centers, words):
        ax.text(k, H0 / 2, word, ha='center', va='center')

    ax.axis('off')
    if fax is None:
        plt.colorbar(h)
        plt.show()


def plot_sentence_grid(words, relevance, fax=None, threshold=10, start_end=[None, None]):
    if fax is None:
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8),
                             gridspec_kw={'height_ratios': [1]})
    else:
        f, ax = fax

    r_normalization = np.max(np.abs(relevance))

    append = [0 for _ in range(threshold - len(words[start_end[0]:start_end[1]]) % threshold)]
    tok_append = ['PAD' for _ in range(len(append))]
    reshape = (-1, threshold)

    R = np.array(relevance[start_end[0]:start_end[1]])  # / r_normalization
    R_plot = np.array(R.tolist() + append)
    sns.heatmap(np.array(R_plot).reshape(reshape),
                annot=np.array(words[start_end[0]:start_end[1]] + tok_append)[np.newaxis, :].reshape(reshape),
                fmt='', ax=ax, cmap='coolwarm', vmin=-r_normalization, vmax=r_normalization,
                annot_kws={"size": 8})
    ax.set_xticks([])
    ax.set_yticks([])
