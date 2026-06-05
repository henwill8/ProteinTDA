from transformers.models.esm.modeling_esmfold import EsmForProteinFoldingOutput
from sidechainnet.dataloaders.SCNProtein import SCNProtein
from sidechainnet_utils import sidechainnet_to_atom37 
import openfold.data.data_transforms as data_transforms
from openfold.utils.loss import AlphaFoldLoss 

def out_conversion(
            esm_out: EsmForProteinFoldingOutput,
            sc_protein: SCNProtein
            ):
    out = {}
    out["sm"] = {}
    out["sm"]["frames"] = esm_out.frames
    out["sm"]["sidechain_frames"] = esm_out.sidechain_frames
    out["sm"]["positions"] = esm_out.positions
    out["sm"]["angles"] = esm_out.angles
    out["sm"]["unnormalized_angles"] = esm_out.unnormalized_angles
    out["tm_logits"] = esm_out.ptm_logits

    # Unsure if this is exactly right
    out["lddt_logits"] = esm_out.lddt_head

    # ? final_atom_positions
    # ? distogram_logits
    # ? experimentally_resolved_logits
    # ? masked_msa_logits

    batch = {}
    # batch from sc_protein


     
    return out, batch
