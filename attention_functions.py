
import re
import requests
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import skimage.transform
import torch
from torch import nn
from torchvision import transforms


## 1k image net classes import
url = "https://raw.githubusercontent.com/xmartlabs/caffeflow/refs/heads/master/examples/imagenet/imagenet-classes.txt"
response = requests.get(url)
assert response.status_code == 200
classes_list = [c.split(',')[0] for c in response.text.split('\n')]
IMAGENET_CLASSES = np.array(classes_list)[:1000]


## Model import
MODEL = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
MODEL.eval()


## Definition of transforms applied to each image
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



## ----- ATTENTION ROLLOUT AND FLOW ----- ## 



class VITAttention:
    """Parent class for both attention rollout and attention flow."""

    def __init__(self, model: nn.Module, head_fusion: str = "mean", discard_ratio: float = 0.9, verbose: bool = False):
        """Initialize the class and create the hooks for attention.

        Args:
            model (Module): The transformer for which we calculate the rollout.
            head_fusion (str): What method to use for multi-head fusion, can either be 'mean', 'max', or 'min'.
            discard_ratio (float): Discard ratio between 0.0 and 1.0.
            verbose (bool): Whether to print informations and tensor shapes for debugging.
        """

        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.verbose = verbose
        self.attentions = []
        self.hooks = []
        
        # We register hooks on all layers in order to access the attentios
        for name, module in self.model.named_modules():
            if "attn.qkv" in name: # On récup la sortie du calcul de Q,K,V car 
                                #recuperer l'attention directement après le dropout ne marche pas
                if self.verbose:
                    print(f"Hook on {name}")
                handle = module.register_forward_hook(self.get_attention)
                self.hooks.append(handle)
    

    def get_attention(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        """Hook function that allows to access raw attention.
        
        We retrieve matrices Q, K and V and compute attention with Q and K
        The code is similar to that of the forward function at 
        https://github.com/facebookresearch/deit/blob/main/models_v2.py#L26
        """

        B, N, C = 1, 197, 192 
        #shape : B = batch size (1), N = nombre de tokens (14x14), C = dimension des tokens
        if self.verbose:
            print(f"Hooked module: {module}")
            print(f"Output Shape : {output.shape}")

        qkv = output.detach().cpu().reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q / 4 
        if self.verbose:
            print(f"Q Shape : {q.shape}")
            print(f"K Shape : {k.shape}")
            print(f"V Shape : {v.shape}")
      
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        if self.verbose:  
            print(f"Attn Shape : {output.shape}")
        self.attentions.append(attn)

    def remove_hooks(self) -> None:
        """Remove the hooks to the model.

        Functions to be called when no longer using the object. 
        If hooks are not removed, it saturates the CPU.
        """

        for handle in self.hooks:
            handle.remove()


## ATTENTION ROLLOUT


class VITAttentionRollout(VITAttention):

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Passes an input through self.model and returns the attention rollout.

        Args:
            input_tensor (Tensor): An input tensor of size 1x3x224x224.

        Returns:
            att_rollout (Tensor): The attention rollout for each token, as a tensor of size 196.
        """

        self.attentions = []
        with torch.no_grad():
            self.model(input_tensor)

        # We now compute the attention rollout from the list of attentions contained in self.attentions

        result = torch.eye(self.attentions[0].size(-1))
        with torch.no_grad():
            for attention in self.attentions:
                if self.head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif self.head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif self.head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"

                # Drop the lowest attentions, but don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*self.discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)

        return result[0, 0, 1:]


def attention_rollout(input_tensor: torch.Tensor, 
                      model: nn.Module, 
                      head_fusion: str = "mean", 
                      discard_ratio: float = 0.0
                      ) -> tuple[torch.Tensor, str]:
    """High-level function doing the attention rollout.

    Args:
        input_tensor (Tensor): The image tensor of size 1x3x224x224 for which we want the attention rollout.
        model (Module): The transformer to be used.
        head_fusion (str): What method to use for multi-head fusion, can either be 'mean', 'max', or 'min'.
        discard_ration (float): Discard ratio between 0.0 and 1.0.
    
    Returns:
        att_rollout (Tensor): The attention rollout for each token, as a tensor of size 196.
        name (str): A name for identification.
    """

    vit_attention_rollout = VITAttentionRollout(model, head_fusion, discard_ratio)
    att_rollout = vit_attention_rollout(input_tensor)
    vit_attention_rollout.remove_hooks()

    name = "attention_rollout_{:.3f}_{}.png".format(discard_ratio, head_fusion)

    return att_rollout, name


class VITAttentionGradRollout(VITAttention):

    def __init__(self, model: nn.Module, head_fusion: str = "mean", discard_ratio: float = 0.9, verbose: bool = False):
        """Initialize the class and create the hooks for attention and gradient.

        Args:
            model (Module): The transformer for which we calculate the rollout.
            head_fusion (str): What method to use for multi-head fusion, can either be 'mean', 'max', or 'min'.
            discard_ratio (float): Discard ratio between 0.0 and 1.0.
            verbose (bool): Whether to print informations and tensor shapes for debugging.
        """
        self.model = model
        self.verbose = verbose
        self.attentions = []
        self.liste_v = []
        self.etage = 0
        self.attention_gradients = []
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.hooks = []
        
        for name, module in self.model.named_modules():
            if ("attn.proj" in name and not "attn.proj_drop" in name):
                handle = module.register_full_backward_hook(self.get_attention_gradient)
                self.hooks.append(handle)
            if "attn.qkv" in name:
                handle = module.register_forward_hook(self.get_attention)
                self.hooks.append(handle)
    
    def get_attention(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Hook qui permet de récupérer les attentions Q,K,V
        On calcule l'attention avec Q et K. 
        On a repris le code de la fonction forward de la classe Attention du modèle
        https://github.com/facebookresearch/deit/blob/main/models_v2.py#L26
        """
        B, N, C = 1, 197, 192 #shape : B = batch size (1), N = nombre de tokens (14x14), C = dimension des tokens
        self.etage+=1 # Ce compteur permet de retrouver l'ordre des V stockés lors du backward
        if self.verbose:
            print(f"Hooked module: {self.etage}")
            print(f"Output Shape : {output.shape}")

        qkv = output.detach().cpu().reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        self.liste_v.append(v) # On stocke les valeurs de V pour les utiliser dans le backward pour reconstruire le gradient
        
        q = q / 4 
        if self.verbose:
            print(f"Q Shape : {q.shape}")
            print(f"K Shape : {k.shape}")
            print(f"V Shape : {v.shape}")
      
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        if self.verbose:  
            print(attn.shape)
            print(f"Attn Shape : {output.shape}")
        self.attentions.append(attn)  

    def get_attention_gradient(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
        """
        Hook qui permet de récupérer le gradient après le softmax de l'attention
        On récupère le gradient à l'input de la projection de l'attention (en aval du softmax),
        puis on remonte le gradient pour obtenir le gradient de l'attention
        https://github.com/facebookresearch/deit/blob/main/models_v2.py#L26
        """
        self.etage -= 1
        if self.verbose:
            print(f"Get gradient of {self.etage}")
            print(f"Input grad shape : {grad_input[0].shape}")
            print(f"Output grad shape : {grad_output[0].shape}")
        self.attention_gradients.append(grad_input[0].cpu().view(1, 197, 12, 16).transpose(1, 2) @ self.liste_v[self.etage].transpose(-2, -1))

    def __call__(self, input_tensor: torch.Tensor, category_index: int):
        """Passes an input through self.model and returns the attention rollout.

        Args:
            input_tensor (Tensor): An input tensor of size 1x3x224x224.
            category_index (int): The category from which we use the gradient. If None, use the predicted category.

        Returns:
            att_rollout (Tensor): The attention rollout for each token, as a tensor of size 196.
        """
        self.attentions = []
        self.attention_gradients = []
        self.model.zero_grad()
        output = self.model(input_tensor)

        # If category index is None, use the predicted category
        if category_index is None:
            category_index = torch.argmax(output[0]).item()

        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        loss = (output*category_mask).sum()
        loss.backward()

        # Now we compute grad rollout
        #return grad_rollout(self.attentions, self.attention_gradients, self.head_fusion, self.discard_ratio)
        result = torch.eye(self.attentions[0].size(-1))
        with torch.no_grad():
            for attention, grad in zip(self.attentions, self.attention_gradients):                
                weights = grad
                if self.head_fusion == "mean":
                    attention_heads_fused = (attention*weights).mean(axis=1)
                elif self.head_fusion == "max":
                    attention_heads_fused = (attention*weights).max(axis=1)[0]
                elif self.head_fusion == "min":
                    attention_heads_fused = (attention*weights).min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"
                # Attention heads can be negative, but we want to keep them positive
                attention_heads_fused[attention_heads_fused < 0] = 0

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*self.discard_ratio), -1, False)
                #indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)

        return result[0, 0, 1:]


def attention_grad_rollout(input_tensor: torch.Tensor, 
                      model: nn.Module, 
                      category_index: int,
                      head_fusion: str = "mean", 
                      discard_ratio: float = 0.0
                      ) -> tuple[torch.Tensor, str]:
    """High-level function doing the attention grad rollout.

    Args:
        input_tensor (Tensor): The image tensor of size 1x3x224x224 for which we want the attention rollout.
        model (Module): The transformer to be used.
        category_index (int): The category from which we use the gradient. If None, use the predicted category.
        head_fusion (str): What method to use for multi-head fusion, can either be 'mean', 'max', or 'min'.
        discard_ration (float): Discard ratio between 0.0 and 1.0.
    
    Returns:
        att_grad_rollout (Tensor): The attention rollout for each token, as a tensor of size 196.
        name (str): A name for identification.
    """

    vit_attention_grad_rollout = VITAttentionGradRollout(model, head_fusion, discard_ratio)
    att_grad_rollout = vit_attention_grad_rollout(input_tensor, category_index)
    vit_attention_grad_rollout.remove_hooks()

    name = "attention_grad_rollout_{:.3f}_{}.png".format(discard_ratio, head_fusion)

    return att_grad_rollout, name


## ATTENTION FLOW


class VITAttentionFlow(VITAttention):

    def __call__(self, input_tensor: torch.Tensor) -> tuple[float, dict]:
        """Passes an input through self.model and returns the attention flow.

        Args:
            input_tensor (Tensor): An input tensor of size 1x3x224x224.

        Returns:
            flow_val (float): la valeur totale du flot maximal
            flow_dict (dict): un dictionnaire décrivant le flot sur chaque arête
        """
        self.attentions = []
        with torch.no_grad():
            self.model(input_tensor)  
        
        # We transform attentions in a graph
        G = self.build_attention_graph(self.attentions)
        L = len(self.attentions)
        N = self.attentions[0].shape[-1]
        # We apply max flow algorihtm
        return self.max_flow_attention(G, L, N)

    def build_attention_graph(self, attentions: list[torch.Tensor]) -> nx.DiGraph:
        """
        Construit un graphe dirigé ACYCLIQUE, dans le sens (l)→(l-1).
        
        Args:
            attentions (list[Tensor]): liste de tenseurs de taille [1, nb_heads, N, N]
                pour chaque couche, où len(attentions)=L.
                Concrètement, attentions[l] = matrice d'attention
                de la couche l, shape = [1, nb_heads, N, N].
                N = nb de tokens, ex. 197 (CLS + patchs).
        
        Returns:
            G (nx.DiGraph):
        """
        G = nx.DiGraph()
        
        # Nombre de couches
        L = len(attentions)
        # Nombre de têtes
        nb_heads = attentions[0].shape[1]
        # Nombre de tokens (CLS + patchs)
        N = attentions[0].shape[-1]

        # Helper pour indexer (layer, token) dans un seul entier
        def node_id(layer, token_idx):
            # On aura des ID distincts pour chaque (l, i)
            return layer*N + token_idx
        
        # On va du plus haut (layer L) vers le plus bas (layer 0)
        # Donc, pour chaque couche l = L,...,1, on crée des arêtes vers l-1
        # (sauf la couche 0 qui n'a pas de couche précédente)
        
        # On itère sur l dans [L, 1] (exclu 0) => en Python range(L, 0, -1)
        # Mais attention, l dans self.attentions est indexé 0..L-1
        # => On peut inverser l'ordre, ou re-mapper...
        
        # Par convention: attentions[0] = couche 0, attentions[1] = couche 1, ...
        # Mais dans l'article, la "couche 1" en code Python correspond à "layer=1".
        # On va boucler 'à l'envers' :
        #   - La "couche L-1" (l'indice L-1) pointe vers L-2,
        #   - etc.
        
        for layer_idx in reversed(range(L)):
            # Fusion des têtes
            # attentions[layer_idx] => shape [1, nb_heads, N, N]
            if self.head_fusion == "mean":
                attn = attentions[layer_idx].mean(dim=1).squeeze(0) # => shape [N, N]
            elif self.head_fusion == "max":
                attn = attentions[layer_idx].max(dim=1)[0].squeeze(0)
            elif self.head_fusion == "min":
                attn = attentions[layer_idx].min(dim=1)[0].squeeze(0)
            else:
                raise ValueError("head_fusion doit être 'mean', 'max' ou 'min'")
            
            # Ajout diagonal (résiduel) + normalisation
            I = torch.eye(N, dtype=attn.dtype, device=attn.device)
            Ares = (attn + I) / 2.0
            row_sum = Ares.sum(dim=-1, keepdim=True)
            row_sum = torch.where(row_sum>1e-9, row_sum, torch.ones_like(row_sum)*1e-9)
            Ares = Ares / row_sum  # shape [N, N]
            
            # On ajoute les arêtes (layer_idx, i) -> (layer_idx-1, j)
            # Mais attention, s'il n'y a pas de "layer_idx - 1" si layer_idx=0
            if layer_idx == 0:
                # pas de couche en-dessous, on ne crée pas d'arêtes
                continue
            
            for i in range(N):
                for j in range(N):
                    w = Ares[i, j].item()
                    if w > 1e-7:
                        # source = (layer_idx, i)
                        # target = (layer_idx-1, j)
                        G.add_edge(
                            node_id(layer_idx, i),
                            node_id(layer_idx-1, j),
                            weight=w
                        )
        
        return G

    def max_flow_attention(self, 
                           G: nx.DiGraph, 
                           L: int, 
                           N: int, 
                           cls_idx: int = 0,
                           ) -> tuple[float, dict]:
        """
        Calcule le flot max depuis (L, cls_idx) = [CLS] de la couche finale
        jusqu'à *tous les tokens* de la couche 0 dans le graphe G,
        en créant un super-sink.

        Args:
            G (DiGraph): le graphe dirigé networkx.DiGraph
            L (int): indice de la dernière couche (si vous avez L+1 couches, c'est la couched'indice L)
            N (int): nombre de tokens par couche (y compris [CLS])
            cls_idx (int): index du token CLS (souvent 0)
    
        Returns:
            flow_val (float): la valeur totale du flot maximal
            flow_dict (dict): un dictionnaire décrivant le flot sur chaque arête
        """

        # 1) On ajoute un super-sink (super puits) vers lequel pointeront
        #    tous les nœuds de la couche 0 (input tokens).
        super_sink = (L+1)*N + 999999  # un ID unique, plus grand que tous les node_id existants
        G.add_node(super_sink)

        # 2) On connecte chaque token de la couche 0 => super_sink
        #    avec une "capacité infinie" (on prend un grand nombre).
        INF_CAP = 1e9
        for token_idx in range(N):
            # Les nœuds de la couche 0 ont un node_id = 0*N + token_idx = token_idx
            G.add_edge(token_idx, super_sink, weight=INF_CAP)

        # 3) La source est (L, cls_idx) => node_id = L*N + cls_idx
        source_idx = (L-1)*N + cls_idx

        # 4) On s'assure qu'aucun poids négatif ou infini n'existe
        for u, v, data in G.edges(data=True):
            w = data["weight"]
            # On remplace si besoin par un petit epsilon
            if w <= 0 or torch.isinf(torch.tensor(w)):
                data["weight"] = 1e-6

        # 5) Appel standard à maximum_flow
        flow_val, flow_dict = nx.maximum_flow(G, source_idx, super_sink, capacity="weight")

        return flow_val, flow_dict
    

def compute_flow_received(flow_dict: dict) -> torch.Tensor:
    """Compute the flow received by each token from a flow dict.

    Args:
        flow_dict (dict): Flow dict returned by max flow attention.

    Returns:
        flow_received (Tensor): Tensor of size 196 containing the flow received by each token.
    """
    # Définition des indices
    layer_1_start = 0  # Premier token de la couche 1
    layer_1_end = 2364  # Dernier token de la couche 1

    flow_received = torch.zeros(196)
    for token in range(1,197):
        total_flux_received = 0
        for node in range(layer_1_start, layer_1_end + 1):
            if node in flow_dict and token in flow_dict[node]:
                total_flux_received += flow_dict[node][token]
        flow_received[token-1] = total_flux_received
    
    return flow_received


def attention_flow(input_tensor: torch.Tensor, 
                   model: nn.Module, 
                   head_fusion: str = "mean", 
                   discard_ratio: float = 0.0
                   ) -> tuple[torch.Tensor, str]:
    """High-level function doing the attention flow algorithm.

    Args:
        input_tensor (Tensor): The image tensor of size 1x3x224x224 for which we want the attention flow.
        model (Module): The transformer to be used.
        head_fusion (str): What method to use for multi-head fusion, can either be 'mean', 'max', or 'min'.
        discard_ration (float): Discard ratio between 0.0 and 1.0.
    
    Returns:
        att_rollout (Tensor): The attention flow received for each token, as a tensor of size 196.
        name (str): A name for identification.
    """

    vit_attention_flow = VITAttentionFlow(model, head_fusion=head_fusion, discard_ratio=discard_ratio)
    flow_val, flow_dict = vit_attention_flow(input_tensor)
    vit_attention_flow.remove_hooks()
    flow_received = compute_flow_received(flow_dict)

    name = "attention_flow_{:.3f}_{}.png".format(discard_ratio, head_fusion)

    return flow_received, name


## ----- INPUT GRADIENTS ----- ##



def input_gradients(model: nn.Module, input_tensor: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Compute the input gradients wrt to each 3x16x16 token.

    Args:
        model (Module): The model to use.
        input_tensor (Tensor): An input tensor of size 1x3x224x224.
    
    Returns:
        target_class (int): The class predicted by the model on the input tensor.
        input_gradients (Tensor): A tensor of size 196 containing the L1 norms of the input gradients wrt to each token.
    """

    model.eval()

    # Enable gradient tracking on input
    input_tensor.requires_grad_(True)

    # Forward pass
    output = model(input_tensor)[0]
    
    # Compute gradients of the predicted class w.r.t. input
    target_class = torch.argmax(output).item()
    output[target_class].backward()  # Backpropagate only for the chosen class
    input_gradients = input_tensor.grad

    # Then we aggregate gradient for each token
    input_gradients = input_gradients.squeeze()
    input_gradients = input_gradients.unfold(1, 16, 16).unfold(2, 16, 16)
    input_gradients = input_gradients.permute(1, 2, 0, 3, 4)
    input_gradients = input_gradients.flatten(0, 1).flatten(1, 3)

    # We finally take the L1 norm, in order to return a tensor of size 196
    input_gradients = torch.sum(abs(input_gradients), dim=-1)
    
    return target_class, input_gradients



## ----- BLANK OUT ----- ##



def blank_out(model: nn.Module, input_tensor: torch.Tensor, method: str = 'avg_color_square') -> torch.Tensor:
    """Applies the blank out method.

    Returns the importance of each of the 196 token. 
    For each 3x16x16 square, its blanked out score is the difference between predicted score
    of the true class for the whole image and score for the same image with the square blanked out.

    Args:
        model (Module):
        input_tensor (Tensor): 
        method (str): The method for blanking out, can either be 'random_noise', 'black_square', 'white_square' or 'avg_color_square'

    Returns:
        blanked_out_score (Tensor): A tensor of size 196 containing the blanked out score of each 3x16x16 token.
    """

    # We first create one image for each token to be blanked out
    # and concatenate them all as a 197x3x224x224 tensor
    # The first image is the original image
    input_tensor = input_tensor.squeeze()
    tens = torch.tile(input_tensor, (197, 1, 1, 1,))
    average_color = torch.mean(input_tensor, dim=(1, 2))[:,None,None]

    for i in range(14):
        for j in range(14):
            # blanked_out_square = tens[14*i+j+1,:,16*i:16*(i+1),16*j:16*(j+1)]

            if method == 'random_noise': # Random noise with values between -1 and 1
                tens[14*i+j+1,:,16*i:16*(i+1),16*j:16*(j+1)] = torch.rand((3, 16, 16)) * 2 -1

            elif method == 'black_square':
                tens[14*i+j+1,:,16*i:16*(i+1),16*j:16*(j+1)] = torch.ones((3, 16, 16), dtype=torch.float) * -1

            elif method == 'white_square':
                tens[14*i+j+1,:,16*i:16*(i+1),16*j:16*(j+1)] = torch.ones((3, 16, 16), dtype=torch.float) * 1
            
            elif method == 'avg_color_square':
                tens[14*i+j+1,:,16*i:16*(i+1),16*j:16*(j+1)] = torch.ones((3, 16, 16), dtype=torch.float) * average_color
                

            else:
                raise NotImplementedError
    
    # Then we do a forward pass
    model.eval()
    with torch.no_grad():
        output = model(tens)
    logits = torch.softmax(output, dim=-1)

    # We determine the true predicted class
    target_class = torch.argmax(logits[0]).item()

    # Then for each square, its blanked_out score is the difference between predicted score
    # of the true class for the whole image and score for the image with the blanked_out square
    blanked_out_score = logits[0, target_class] - logits[1:, target_class]

    # A negative score means that the model can do better prediction without the pixel
    # We cap the negative values to 0
    blanked_out_score = blanked_out_score * (blanked_out_score >= 0)

    return blanked_out_score



## ----- VISUALIZATION FUNCTIONS ----- ##



def create_mask(t: torch.Tensor) -> np.ndarray:
    """Transforms a 1d tensor into a 2d mask array that can then be applied for image visualization.

    Args:
        t (Tensor): A tensor of size 196.
    
    Returns:
        ndarray: The mask array of size 14x14.
    """
    t = t.detach()
    width = int(t.size(-1)**0.5)
    mask = t.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def show_mask_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shows the mask on the image.

    The mask is eventually reshaped to the img size.

    Args:
        img (ndarray): The image as a 3 channels array.
        mask (ndarray):
    
    Returns:
        ndarray: The masked image.   
    """

    mask = skimage.transform.resize(mask, (img.shape[0], img.shape[1]), anti_aliasing=True)
    img = np.float32(img) / 255  # Normalisation de l'image
    heatmap = plt.cm.jet(mask)[:, :, :3]  # Conversion du masque en colormap (sans canal alpha)
    heatmap = np.float32(heatmap)
    cam = heatmap + img  # Fusion de l'image et du masque
    cam = cam / np.max(cam)  # Normalisation
    return np.uint8(255 * cam)  # Conversion en image 8 bits


def display_output(output: torch.Tensor) -> None:
    """Displays the output of a model.

    Plots the 10 most likely classes and their estimated probabilities.

    Args:
        output (Tensor): A tensor of size 1000, corresponding to the 1k classes.
    """

    output = output.detach().squeeze()
    assert output.shape == (1000,)

    softmax_output = torch.softmax(output, dim=0)
    sorted_probs, sorted_indices = torch.sort(softmax_output, dim=0, descending=False)

    plt.barh(range(10), sorted_probs[990:])
    plt.yticks(range(10), ['{} ({})'.format(IMAGENET_CLASSES[i], i) for i in sorted_indices[990:]])
    plt.xlabel('Softmax prediction')
    plt.grid(axis='x', linestyle='--', color='black')

    plt.show()


## ----- STATISTICAL ANALYSIS ----- ##

def spearman_correlation(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Calcule la corrélation de Spearman entre deux tenseurs 1D de même taille.

    Args:
        tensor1 (Tensor):
        tensor2 (Tensor):

    Returns:
        spearman_corr (float): corrélation de Spearman entre les deux tenseurs
    """
    assert tensor1.shape == tensor2.shape, "Les deux tenseurs doivent avoir la même taille"

    tensor1 = tensor1.flatten().detach().numpy()
    tensor2 = tensor2.flatten().detach().numpy()

    corr, _ = scipy.stats.spearmanr(tensor1, tensor2)
    return corr.item()


def get_image_list(n: int = None):
    """Returns the list of 1000 images from the ImageNet dataset.
    Images are fetched from the github repo. It contains one image per classes.
    https://github.com/EliSchwartz/imagenet-sample-images/tree/master
    The time to request each image can be quite long and it is advised to fetch only a part.
    The returned image are all resized to be of size 224x224.

    Args:
        n (int|None): Amount of images to fetch. If None fetch all images.

    Returns:
        images (list[Image]):    
    """

    # First we request the gallery.md, which contains link to all 1000 images
    gallery_url = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/refs/heads/master/gallery.md"
    gallery_text = requests.get(gallery_url).text
    # We use regex as we have to find the links in the text.
    files_list = re.findall(r"n\d*_\D*\.JPEG", gallery_text)

    if n:
        files_list = files_list[:n]

    images = []

    url_base = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/refs/heads/master/"
    for filename in files_list:
        response = requests.get(url_base + filename)
        image = Image.open(BytesIO(response.content))
        image = image.resize((224, 224))
        images.append(image)

    return images