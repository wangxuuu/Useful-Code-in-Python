class Channels():
    def PowerNormalize(self, x):
        x_square = torch.mul(x, x)
        power = math.sqrt(2) * x_square.mean(dim=1, keepdim=True).sqrt()
        out = torch.div(x, power)
        return out

    def SNR_to_noise(self, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / ((2 * snr)**0.5)
        return noise_std       
        
    def AWGN(self, Tx_sig, snr):
        n_var = self.SNR_to_noise(snr)
        Rx_sig = self.PowerNormalize(Tx_sig)
        Rx_sig = Rx_sig + torch.normal(0, n_var, size=Rx_sig.shape, device=device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, snr):
        shape = Tx_sig.shape # (B, 2M)
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        n_var = self.SNR_to_noise(snr)
        Tx_sig = self.PowerNormalize(Tx_sig)

        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H) # (B, M, 2) (2, 2)
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape, device=device)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, snr, K_dB=0):
        shape = Tx_sig.shape
        K = 10**(K_dB/10)
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1)) * math.sqrt(1/2)
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(0, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        n_var = self.SNR_to_noise(snr)
        Tx_sig = self.PowerNormalize(Tx_sig)

        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape, device=device)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig
