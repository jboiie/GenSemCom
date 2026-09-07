import os
import numpy as np


def get_project_paths():
    """Return useful paths: project root, data dir, and key files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    embeddings_npy_path = os.path.join(data_dir, "minilm_embeddings.npy")
    quantized_npy_path = os.path.join(data_dir, "minilm_quantized.npy")
    rec_embeddings_npy_path = os.path.join(data_dir, "minilm_embeddings_reconstructed.npy")
    stats_npy_path = os.path.join(data_dir, "minilm_quant_stats.npz")  # stores emin/emax

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "embeddings": embeddings_npy_path,
        "quantized": quantized_npy_path,
        "reconstructed": rec_embeddings_npy_path,
        "stats": stats_npy_path,
    }


# ---------- Quantization / Dequantization ----------


def compute_quantization_stats(embeddings: np.ndarray):
    """
    Compute per-dimension min and max over the entire dataset.
    embeddings: shape (N, D)
    Returns emin, emax each shape (D,)
    """
    emin = embeddings.min(axis=0)
    emax = embeddings.max(axis=0)
    return emin, emax


def quantize_embeddings(embeddings: np.ndarray, emin: np.ndarray, emax: np.ndarray):
    """
    Uniform 8-bit per-dimension quantization.
    Returns uint8 array Q with same shape as embeddings.
    """
    eps = 1e-8
    # Normalize to [0, 1]
    norm = (embeddings - emin) / (emax - emin + eps)
    norm = np.clip(norm, 0.0, 1.0)
    # Map to [0, 255]
    Q = np.rint(norm * 255.0).astype(np.uint8)
    return Q


def dequantize_embeddings(Q: np.ndarray, emin: np.ndarray, emax: np.ndarray):
    """
    Inverse of quantize_embeddings.
    Q: uint8 array of shape (N, D)
    Returns float array approximating original embeddings.
    """
    norm = Q.astype(np.float32) / 255.0
    rec = norm * (emax - emin) + emin
    return rec


# ---------- Bit packing / unpacking ----------


def pack_bits_from_uint8(Q: np.ndarray) -> np.ndarray:
    """
    Convert uint8 matrix Q (N, D) to a 1D bit array of 0/1.
    Uses numpy.unpackbits for efficiency.
    """
    Q_flat = Q.reshape(-1)  # (N*D,)
    bits = np.unpackbits(Q_flat)  # shape (N*D*8,)
    return bits.astype(np.int8)


def unpack_bits_to_uint8(bits: np.ndarray, original_shape) -> np.ndarray:
    """
    Convert 1D bits back to uint8 matrix Q with original_shape (N, D).
    """
    # Ensure length is multiple of 8
    assert bits.size % 8 == 0, "Bit array length must be multiple of 8."
    uint8_vals = np.packbits(bits.astype(np.uint8))  # 1D uint8
    Q_rec = uint8_vals.reshape(original_shape).astype(np.uint8)
    return Q_rec


# ---------- BPSK modulation / demodulation ----------


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Map bits {0,1} to BPSK symbols {-1.0, +1.0}.
    bits: 1D array of 0/1
    """
    return 2.0 * bits.astype(np.float32) - 1.0  # 0 -> -1, 1 -> +1


def bpsk_demodulate(received: np.ndarray) -> np.ndarray:
    """
    Hard-decision BPSK demodulation: r >= 0 -> 1, else 0.
    """
    bits_rec = (received >= 0).astype(np.int8)
    return bits_rec


# ---------- Rayleigh + AWGN channel for BPSK ----------


def rayleigh_fading(num_symbols: int) -> np.ndarray:
    """
    Generate complex Rayleigh fading coefficients h ~ CN(0,1)
    for each symbol.
    """
    h_real = np.random.normal(0.0, 1.0 / np.sqrt(2.0), size=num_symbols)
    h_imag = np.random.normal(0.0, 1.0 / np.sqrt(2.0), size=num_symbols)
    h = h_real + 1j * h_imag
    return h.astype(np.complex64)


def rayleigh_awgn_channel_bpsk(symbols: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply flat Rayleigh fading + AWGN to real BPSK symbols.

    symbols: 1D array of real BPSK symbols in {-1,+1}
    snr_db:  SNR per symbol in dB (same definition as AWGN case)

    Returns:
        h      : complex Rayleigh fading coefficients (per symbol)
        s_hat  : equalized noisy symbol estimates (real-valued)
    """
    # 1. Rayleigh fading coefficients
    num_symbols = symbols.size
    h = rayleigh_fading(num_symbols)  # shape (N,)

    # 2. AWGN noise (complex) for given SNR
    snr_linear = 10.0 ** (snr_db / 10.0)
    # For BPSK with |s|^2 = 1, symbol energy Es = 1
    noise_variance = 1.0 / (2.0 * snr_linear)  # per complex symbol
    noise_std = np.sqrt(noise_variance)

    n_real = np.random.normal(0.0, noise_std / np.sqrt(2.0), size=num_symbols)
    n_imag = np.random.normal(0.0, noise_std / np.sqrt(2.0), size=num_symbols)
    n = (n_real + 1j * n_imag).astype(np.complex64)

    # 3. Channel: r = h * s + n
    s_complex = symbols.astype(np.complex64)
    r = h * s_complex + n

    # 4. Equalization: s_hat = conj(h) * r / |h|^2
    eps = 1e-12
    s_hat_complex = np.conj(h) * r / (np.abs(h) ** 2 + eps)

    # Only the real part is needed for BPSK decisions
    s_hat = s_hat_complex.real.astype(np.float32)
    return h, s_hat


# ---------- High-level pipeline functions ----------


def run_channel_for_snr(snr_db: float):
    """
    Full pipeline for a single SNR value:
    load embeddings -> quantize -> pack bits -> BPSK + Rayleigh+AWGN ->
    demodulate -> unpack bits -> dequantize -> save reconstructed embeddings.
    Also returns original and reconstructed embeddings for evaluation.
    """
    paths = get_project_paths()

    # 1. Load original embeddings
    embeddings = np.load(paths["embeddings"])  # shape (N, D)
    N, D = embeddings.shape
    print(f"[SNR {snr_db} dB] Loaded embeddings, shape: {embeddings.shape}")

    # 2. Compute or load quantization stats
    emin, emax = compute_quantization_stats(embeddings)

    # 3. Quantize
    Q = quantize_embeddings(embeddings, emin, emax)
    np.save(paths["quantized"], Q)
    np.savez(paths["stats"], emin=emin, emax=emax)

    # 4. Pack to bits
    bits = pack_bits_from_uint8(Q)
    print(f"[SNR {snr_db} dB] Bitstream length: {bits.size}")

    # 5. BPSK + Rayleigh + AWGN channel
    tx_symbols = bpsk_modulate(bits)  # real BPSK symbols
    _, rx_eq = rayleigh_awgn_channel_bpsk(tx_symbols, snr_db=snr_db)

    # 6. Demodulate after equalization
    bits_rec = bpsk_demodulate(rx_eq)

    # 7. Unpack bits back to uint8 matrix
    Q_rec = unpack_bits_to_uint8(bits_rec, original_shape=(N, D))

    # 8. Dequantize to reconstructed embeddings
    rec_embeddings = dequantize_embeddings(Q_rec, emin, emax)
    np.save(paths["reconstructed"], rec_embeddings)
    print(f"[SNR {snr_db} dB] Reconstructed embeddings saved to: {paths['reconstructed']}")

    return embeddings, rec_embeddings


def main():
    """
    Example usage: run the channel for a list of SNR values.
    Later, you can call this from run_pipeline.py or evaluate.py.
    """
    snr_values_db = [0, 5, 10, 15, 20]
    for snr in snr_values_db:
        print("=" * 60)
        print(f"Running channel simulation for SNR = {snr} dB")
        embeddings, rec_embeddings = run_channel_for_snr(snr)
        print(f"Done for SNR {snr} dB. Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
