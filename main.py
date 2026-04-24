import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SSFN_MODEL_DIR = os.path.join(BASE_DIR, "ssfn_model")
DEFAULT_TOKENIZER_DIR = os.path.join(BASE_DIR, "ssfn_tokenizer")
DEFAULT_BPE_CODES_PATH = os.path.join(BASE_DIR, "protein_codes_rdrp_20000.txt")
DEFAULT_ESM_MODEL_NAME = "esm2_t36_3B_UR50D"


def build_parser():  # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        description="Predict the probability that a protein sequence is a viral RdRP."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--sequence", help="Protein sequence to predict.")
    input_group.add_argument(
        "--sequence-file",
        type=str,
        help="Path to a text file containing a protein sequence.",
    )
    parser.add_argument(
        "--protein-id",
        default="protein_1",
        help="Protein identifier used by the ESM converter.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Computation device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--ssfn-model-dir",
        type=str,
        default=DEFAULT_SSFN_MODEL_DIR,
        help="Path to the SSFN model directory.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=DEFAULT_TOKENIZER_DIR,
        help="Path to the tokenizer directory.",
    )
    parser.add_argument(
        "--bpe-codes-path",
        type=str,
        default=DEFAULT_BPE_CODES_PATH,
        help="Path to the BPE codes file.",
    )
    parser.add_argument(
        "--esm-model-name",
        default=DEFAULT_ESM_MODEL_NAME,
        help="ESM model name to load via fair-esm.",
    )
    return parser


def load_sequence(args):  # type: (argparse.Namespace) -> str
    if args.sequence is not None:
        return args.sequence.strip()
    with open(args.sequence_file, encoding="utf-8") as f:
        return f.read().strip()


def main():  # type: () -> None
    parser = build_parser()
    args = parser.parse_args()

    from lucaprot_predictor import LucaProtPredictor

    sequence = load_sequence(args)
    predictor = LucaProtPredictor(
        ssfn_model_dir=args.ssfn_model_dir,
        tokenizer_dir=args.tokenizer_dir,
        bpe_codes_path=args.bpe_codes_path,
        device=args.device,
        esm_model_name=args.esm_model_name,
    )
    probability = predictor.predict(sequence, prot_id=args.protein_id)

    print(f"Protein ID: {args.protein_id}")
    print(f"Probability of being a viral RdRP: {probability:.6f}")


if __name__ == "__main__":
    main()
