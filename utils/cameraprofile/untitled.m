

ToneCurve = ToneCurves(75, :);
N = length(ToneCurve);
Curve = reshape(ToneCurve, 2, N / 2);
x = Curve(1, :);
y = Curve(2, :);
delta = 1.0e-6;
xi = 0 : delta : 1.0;
yi = interp1(x, y, xi, 'spline');
yi_inv = interp1(yi, xi, xi, 'spline');

yi = single(yi);
yi_inv = single(yi_inv);
