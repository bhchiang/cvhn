real = [-5:0.01:5];
imag = [-5:0.01:5];

[Real, Imag] = meshgrid(real,imag);
Real_ReLU = 1.0*(Real > 0);
Complex_ReLU = 1.0*(Real > 0).*(Imag > 0);
Complex_Cardioid = 0.5*(1 + Real ./ sqrt(Real.^2 + Imag.^2));
b = -1;
modReLU = max(0,(sqrt(Real.^2 + Imag.^2)+b))./sqrt(Real.^2 + Imag.^2);

% Plot Modulation
figure('Renderer', 'painters','Position', [0 0 3*230 3*180]);
subplot(2, 2, 1);
imagesc(real, imag, Real_ReLU);colorbar;caxis([0, 1]);
set(gca,'YDir','normal');
title('Real ReLU');xlabel('Real');ylabel('Imaginary');
subplot(2, 2, 2);
imagesc(real, imag, Complex_ReLU);colorbar;caxis([0, 1]);
set(gca,'YDir','normal');
title('Complex ReLU');xlabel('Real');ylabel('Imaginary');
subplot(2, 2, 3);
imagesc(real, imag, Complex_Cardioid);colorbar;caxis([0, 1]);
set(gca,'YDir','normal');
title('Complex Cardioid');xlabel('Real');ylabel('Imaginary');
subplot(2, 2, 4);
imagesc(real, imag, modReLU);colorbar;caxis([0, 1]);
set(gca,'YDir','normal');
title('modReLU [3]');xlabel('Real');ylabel('Imaginary');
