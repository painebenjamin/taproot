from __future__ import annotations

from taproot.util import PretrainedModelMixin, get_added_token_dict

from typing import Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import LlamaTokenizerFast # type: ignore[import-untyped]

__all__ = [
    "LlavaLlamaTokenizer"
]

class LlavaLlamaTokenizer(PretrainedModelMixin):
    """
    Tokenizer for LLaVA/LLaMA
    """
    init_file_urls = {
        "tokenizer_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-llava-llama-tokenizer-vocab.json",
        "special_tokens_map_file": "https://huggingface.co/benjamin-paine/taproot-common/resolve/main/text-encoding-llava-llama-tokenizer-special-tokens-map.json",
    }

    @classmethod
    def get_model_class(cls) -> Type[LlamaTokenizerFast]:
        """
        Get the model class for the FLUX Tokenizer.
        """
        from transformers import LlamaTokenizerFast
        return LlamaTokenizerFast # type: ignore[no-any-return]

    @classmethod
    def get_default_config(cls) -> Optional[Dict[str, Any]]:
        """
        et the default configuration for the FLUX Tokenizer.
        """
        return {
            "add_bos_token": True,
            "add_eos_token": False,
            "add_prefix_space": None,
            "added_tokens_decoder": get_added_token_dict({
                "128000": {
                    "content": "<|begin_of_text|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128001": {
                    "content": "<|end_of_text|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128002": {
                    "content": "<|reserved_special_token_0|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128003": {
                    "content": "<|reserved_special_token_1|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128004": {
                    "content": "<|reserved_special_token_2|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128005": {
                    "content": "<|reserved_special_token_3|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128006": {
                    "content": "<|start_header_id|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128007": {
                    "content": "<|end_header_id|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128008": {
                    "content": "<|reserved_special_token_4|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128009": {
                    "content": "<|eot_id|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128010": {
                    "content": "<|reserved_special_token_5|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128011": {
                    "content": "<|reserved_special_token_6|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128012": {
                    "content": "<|reserved_special_token_7|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128013": {
                    "content": "<|reserved_special_token_8|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128014": {
                    "content": "<|reserved_special_token_9|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128015": {
                    "content": "<|reserved_special_token_10|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128016": {
                    "content": "<|reserved_special_token_11|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128017": {
                    "content": "<|reserved_special_token_12|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128018": {
                    "content": "<|reserved_special_token_13|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128019": {
                    "content": "<|reserved_special_token_14|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128020": {
                    "content": "<|reserved_special_token_15|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128021": {
                    "content": "<|reserved_special_token_16|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128022": {
                    "content": "<|reserved_special_token_17|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128023": {
                    "content": "<|reserved_special_token_18|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128024": {
                    "content": "<|reserved_special_token_19|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128025": {
                    "content": "<|reserved_special_token_20|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128026": {
                    "content": "<|reserved_special_token_21|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128027": {
                    "content": "<|reserved_special_token_22|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128028": {
                    "content": "<|reserved_special_token_23|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128029": {
                    "content": "<|reserved_special_token_24|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128030": {
                    "content": "<|reserved_special_token_25|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128031": {
                    "content": "<|reserved_special_token_26|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128032": {
                    "content": "<|reserved_special_token_27|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128033": {
                    "content": "<|reserved_special_token_28|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128034": {
                    "content": "<|reserved_special_token_29|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128035": {
                    "content": "<|reserved_special_token_30|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128036": {
                    "content": "<|reserved_special_token_31|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128037": {
                    "content": "<|reserved_special_token_32|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128038": {
                    "content": "<|reserved_special_token_33|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128039": {
                    "content": "<|reserved_special_token_34|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128040": {
                    "content": "<|reserved_special_token_35|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128041": {
                    "content": "<|reserved_special_token_36|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128042": {
                    "content": "<|reserved_special_token_37|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128043": {
                    "content": "<|reserved_special_token_38|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128044": {
                    "content": "<|reserved_special_token_39|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128045": {
                    "content": "<|reserved_special_token_40|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128046": {
                    "content": "<|reserved_special_token_41|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128047": {
                    "content": "<|reserved_special_token_42|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128048": {
                    "content": "<|reserved_special_token_43|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128049": {
                    "content": "<|reserved_special_token_44|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128050": {
                    "content": "<|reserved_special_token_45|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128051": {
                    "content": "<|reserved_special_token_46|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128052": {
                    "content": "<|reserved_special_token_47|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128053": {
                    "content": "<|reserved_special_token_48|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128054": {
                    "content": "<|reserved_special_token_49|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128055": {
                    "content": "<|reserved_special_token_50|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128056": {
                    "content": "<|reserved_special_token_51|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128057": {
                    "content": "<|reserved_special_token_52|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128058": {
                    "content": "<|reserved_special_token_53|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128059": {
                    "content": "<|reserved_special_token_54|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128060": {
                    "content": "<|reserved_special_token_55|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128061": {
                    "content": "<|reserved_special_token_56|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128062": {
                    "content": "<|reserved_special_token_57|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128063": {
                    "content": "<|reserved_special_token_58|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128064": {
                    "content": "<|reserved_special_token_59|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128065": {
                    "content": "<|reserved_special_token_60|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128066": {
                    "content": "<|reserved_special_token_61|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128067": {
                    "content": "<|reserved_special_token_62|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128068": {
                    "content": "<|reserved_special_token_63|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128069": {
                    "content": "<|reserved_special_token_64|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128070": {
                    "content": "<|reserved_special_token_65|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128071": {
                    "content": "<|reserved_special_token_66|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128072": {
                    "content": "<|reserved_special_token_67|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128073": {
                    "content": "<|reserved_special_token_68|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128074": {
                    "content": "<|reserved_special_token_69|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128075": {
                    "content": "<|reserved_special_token_70|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128076": {
                    "content": "<|reserved_special_token_71|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128077": {
                    "content": "<|reserved_special_token_72|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128078": {
                    "content": "<|reserved_special_token_73|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128079": {
                    "content": "<|reserved_special_token_74|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128080": {
                    "content": "<|reserved_special_token_75|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128081": {
                    "content": "<|reserved_special_token_76|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128082": {
                    "content": "<|reserved_special_token_77|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128083": {
                    "content": "<|reserved_special_token_78|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128084": {
                    "content": "<|reserved_special_token_79|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128085": {
                    "content": "<|reserved_special_token_80|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128086": {
                    "content": "<|reserved_special_token_81|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128087": {
                    "content": "<|reserved_special_token_82|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128088": {
                    "content": "<|reserved_special_token_83|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128089": {
                    "content": "<|reserved_special_token_84|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128090": {
                    "content": "<|reserved_special_token_85|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128091": {
                    "content": "<|reserved_special_token_86|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128092": {
                    "content": "<|reserved_special_token_87|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128093": {
                    "content": "<|reserved_special_token_88|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128094": {
                    "content": "<|reserved_special_token_89|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128095": {
                    "content": "<|reserved_special_token_90|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128096": {
                    "content": "<|reserved_special_token_91|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128097": {
                    "content": "<|reserved_special_token_92|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128098": {
                    "content": "<|reserved_special_token_93|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128099": {
                    "content": "<|reserved_special_token_94|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128100": {
                    "content": "<|reserved_special_token_95|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128101": {
                    "content": "<|reserved_special_token_96|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128102": {
                    "content": "<|reserved_special_token_97|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128103": {
                    "content": "<|reserved_special_token_98|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128104": {
                    "content": "<|reserved_special_token_99|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128105": {
                    "content": "<|reserved_special_token_100|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128106": {
                    "content": "<|reserved_special_token_101|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128107": {
                    "content": "<|reserved_special_token_102|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128108": {
                    "content": "<|reserved_special_token_103|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128109": {
                    "content": "<|reserved_special_token_104|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128110": {
                    "content": "<|reserved_special_token_105|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128111": {
                    "content": "<|reserved_special_token_106|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128112": {
                    "content": "<|reserved_special_token_107|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128113": {
                    "content": "<|reserved_special_token_108|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128114": {
                    "content": "<|reserved_special_token_109|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128115": {
                    "content": "<|reserved_special_token_110|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128116": {
                    "content": "<|reserved_special_token_111|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128117": {
                    "content": "<|reserved_special_token_112|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128118": {
                    "content": "<|reserved_special_token_113|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128119": {
                    "content": "<|reserved_special_token_114|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128120": {
                    "content": "<|reserved_special_token_115|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128121": {
                    "content": "<|reserved_special_token_116|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128122": {
                    "content": "<|reserved_special_token_117|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128123": {
                    "content": "<|reserved_special_token_118|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128124": {
                    "content": "<|reserved_special_token_119|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128125": {
                    "content": "<|reserved_special_token_120|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128126": {
                    "content": "<|reserved_special_token_121|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128127": {
                    "content": "<|reserved_special_token_122|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128128": {
                    "content": "<|reserved_special_token_123|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128129": {
                    "content": "<|reserved_special_token_124|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128130": {
                    "content": "<|reserved_special_token_125|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128131": {
                    "content": "<|reserved_special_token_126|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128132": {
                    "content": "<|reserved_special_token_127|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128133": {
                    "content": "<|reserved_special_token_128|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128134": {
                    "content": "<|reserved_special_token_129|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128135": {
                    "content": "<|reserved_special_token_130|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128136": {
                    "content": "<|reserved_special_token_131|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128137": {
                    "content": "<|reserved_special_token_132|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128138": {
                    "content": "<|reserved_special_token_133|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128139": {
                    "content": "<|reserved_special_token_134|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128140": {
                    "content": "<|reserved_special_token_135|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128141": {
                    "content": "<|reserved_special_token_136|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128142": {
                    "content": "<|reserved_special_token_137|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128143": {
                    "content": "<|reserved_special_token_138|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128144": {
                    "content": "<|reserved_special_token_139|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128145": {
                    "content": "<|reserved_special_token_140|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128146": {
                    "content": "<|reserved_special_token_141|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128147": {
                    "content": "<|reserved_special_token_142|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128148": {
                    "content": "<|reserved_special_token_143|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128149": {
                    "content": "<|reserved_special_token_144|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128150": {
                    "content": "<|reserved_special_token_145|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128151": {
                    "content": "<|reserved_special_token_146|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128152": {
                    "content": "<|reserved_special_token_147|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128153": {
                    "content": "<|reserved_special_token_148|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128154": {
                    "content": "<|reserved_special_token_149|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128155": {
                    "content": "<|reserved_special_token_150|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128156": {
                    "content": "<|reserved_special_token_151|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128157": {
                    "content": "<|reserved_special_token_152|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128158": {
                    "content": "<|reserved_special_token_153|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128159": {
                    "content": "<|reserved_special_token_154|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128160": {
                    "content": "<|reserved_special_token_155|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128161": {
                    "content": "<|reserved_special_token_156|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128162": {
                    "content": "<|reserved_special_token_157|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128163": {
                    "content": "<|reserved_special_token_158|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128164": {
                    "content": "<|reserved_special_token_159|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128165": {
                    "content": "<|reserved_special_token_160|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128166": {
                    "content": "<|reserved_special_token_161|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128167": {
                    "content": "<|reserved_special_token_162|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128168": {
                    "content": "<|reserved_special_token_163|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128169": {
                    "content": "<|reserved_special_token_164|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128170": {
                    "content": "<|reserved_special_token_165|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128171": {
                    "content": "<|reserved_special_token_166|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128172": {
                    "content": "<|reserved_special_token_167|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128173": {
                    "content": "<|reserved_special_token_168|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128174": {
                    "content": "<|reserved_special_token_169|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128175": {
                    "content": "<|reserved_special_token_170|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128176": {
                    "content": "<|reserved_special_token_171|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128177": {
                    "content": "<|reserved_special_token_172|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128178": {
                    "content": "<|reserved_special_token_173|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128179": {
                    "content": "<|reserved_special_token_174|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128180": {
                    "content": "<|reserved_special_token_175|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128181": {
                    "content": "<|reserved_special_token_176|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128182": {
                    "content": "<|reserved_special_token_177|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128183": {
                    "content": "<|reserved_special_token_178|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128184": {
                    "content": "<|reserved_special_token_179|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128185": {
                    "content": "<|reserved_special_token_180|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128186": {
                    "content": "<|reserved_special_token_181|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128187": {
                    "content": "<|reserved_special_token_182|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128188": {
                    "content": "<|reserved_special_token_183|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128189": {
                    "content": "<|reserved_special_token_184|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128190": {
                    "content": "<|reserved_special_token_185|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128191": {
                    "content": "<|reserved_special_token_186|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128192": {
                    "content": "<|reserved_special_token_187|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128193": {
                    "content": "<|reserved_special_token_188|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128194": {
                    "content": "<|reserved_special_token_189|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128195": {
                    "content": "<|reserved_special_token_190|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128196": {
                    "content": "<|reserved_special_token_191|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128197": {
                    "content": "<|reserved_special_token_192|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128198": {
                    "content": "<|reserved_special_token_193|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128199": {
                    "content": "<|reserved_special_token_194|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128200": {
                    "content": "<|reserved_special_token_195|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128201": {
                    "content": "<|reserved_special_token_196|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128202": {
                    "content": "<|reserved_special_token_197|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128203": {
                    "content": "<|reserved_special_token_198|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128204": {
                    "content": "<|reserved_special_token_199|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128205": {
                    "content": "<|reserved_special_token_200|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128206": {
                    "content": "<|reserved_special_token_201|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128207": {
                    "content": "<|reserved_special_token_202|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128208": {
                    "content": "<|reserved_special_token_203|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128209": {
                    "content": "<|reserved_special_token_204|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128210": {
                    "content": "<|reserved_special_token_205|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128211": {
                    "content": "<|reserved_special_token_206|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128212": {
                    "content": "<|reserved_special_token_207|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128213": {
                    "content": "<|reserved_special_token_208|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128214": {
                    "content": "<|reserved_special_token_209|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128215": {
                    "content": "<|reserved_special_token_210|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128216": {
                    "content": "<|reserved_special_token_211|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128217": {
                    "content": "<|reserved_special_token_212|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128218": {
                    "content": "<|reserved_special_token_213|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128219": {
                    "content": "<|reserved_special_token_214|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128220": {
                    "content": "<|reserved_special_token_215|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128221": {
                    "content": "<|reserved_special_token_216|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128222": {
                    "content": "<|reserved_special_token_217|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128223": {
                    "content": "<|reserved_special_token_218|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128224": {
                    "content": "<|reserved_special_token_219|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128225": {
                    "content": "<|reserved_special_token_220|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128226": {
                    "content": "<|reserved_special_token_221|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128227": {
                    "content": "<|reserved_special_token_222|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128228": {
                    "content": "<|reserved_special_token_223|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128229": {
                    "content": "<|reserved_special_token_224|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128230": {
                    "content": "<|reserved_special_token_225|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128231": {
                    "content": "<|reserved_special_token_226|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128232": {
                    "content": "<|reserved_special_token_227|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128233": {
                    "content": "<|reserved_special_token_228|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128234": {
                    "content": "<|reserved_special_token_229|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128235": {
                    "content": "<|reserved_special_token_230|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128236": {
                    "content": "<|reserved_special_token_231|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128237": {
                    "content": "<|reserved_special_token_232|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128238": {
                    "content": "<|reserved_special_token_233|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128239": {
                    "content": "<|reserved_special_token_234|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128240": {
                    "content": "<|reserved_special_token_235|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128241": {
                    "content": "<|reserved_special_token_236|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128242": {
                    "content": "<|reserved_special_token_237|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128243": {
                    "content": "<|reserved_special_token_238|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128244": {
                    "content": "<|reserved_special_token_239|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128245": {
                    "content": "<|reserved_special_token_240|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128246": {
                    "content": "<|reserved_special_token_241|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128247": {
                    "content": "<|reserved_special_token_242|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128248": {
                    "content": "<|reserved_special_token_243|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128249": {
                    "content": "<|reserved_special_token_244|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128250": {
                    "content": "<|reserved_special_token_245|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128251": {
                    "content": "<|reserved_special_token_246|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128252": {
                    "content": "<|reserved_special_token_247|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128253": {
                    "content": "<|reserved_special_token_248|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128254": {
                    "content": "<|reserved_special_token_249|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128255": {
                    "content": "<|reserved_special_token_250|>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128256": {
                    "content": "<unk>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128257": {
                    "content": "<image>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                },
                "128258": {
                    "content": "<pad>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                }
            }),
            "bos_token": "<|begin_of_text|>",
            "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}",
            "clean_up_tokenization_spaces": True,
            "eos_token": "<|end_of_text|>",
            "extra_special_tokens": {},
            "legacy": True,
            "model_input_names": [
                "input_ids",
                "attention_mask"
            ],
            "model_max_length": 1e+30,
            "pad_token": "<pad>",
            "padding_side": "right",
            "processor_class": "LlavaProcessor",
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "<unk>",
            "use_default_system_prompt": False
        }
