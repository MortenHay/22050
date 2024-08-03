from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gain(G):
    return 20 * log(abs(G), 10)


def phase(G):
    return (
        180.0
        / pi
        * (
            atan2(im(fraction(G)[0]), re(fraction(G)[0]))
            - atan2(im(fraction(G)[1]), re(fraction(G)[1]))
        )
    )


def sallenKeyBode(b, a, plot_range=(-2, 3)):
    j = I
    w = Symbol("omega", real=True)
    jw = j * w
    s = Symbol("s")
    num = sum([b[i] * s ** (len(b) - i - 1) for i in range(len(b))])
    zero = roots(num, s)
    numF = prod([(s - s0) ** zero[s0] for s0 in zero])
    numM = simplify((num / numF).subs(s, 1))

    den = sum([a[i] * s ** (len(a) - i - 1) for i in range(len(a))])
    poles = roots(den, s)
    denF = prod([(s - s0) ** poles[s0] for s0 in poles])
    denM = simplify((den / denF).subs(s, 1))

    K = numM / denM
    H = 1
    C = K

    Os = []
    Ls = []
    Qs = []

    for zer in zero:
        if zer == 0:
            H *= s ** zero[zer]
            Os.append(s ** zero[zer])
        elif zer.is_real:
            H *= (s / zer + 1) ** zero[zer]
            C *= zer ** zero[zer]
            Ls.append((-s / zer + 1) ** zero[zer])
        else:
            if im(zer) > 0:
                Q = (s + zer) * (s + zer.conjugate())
                q0 = (Q.subs(s, 0)).simplify()
                q1 = (Q.simplify()).coeff(s)
                H *= (s**2 / q0 + q1 / q0 * s + 1) ** zero[zer]
                C *= q0 ** zero[zer]
                Qs.append((s**2 / q0 + q1 / q0 * s + 1) ** zero[zer])

    for pole in poles:
        if pole == 0:
            H /= s
            Os.append(s ** (-poles[pole]))
        elif pole.is_real:
            H /= (-s / pole + 1) ** poles[pole]
            C /= pole ** poles[pole]
            Ls.append((-s / pole + 1) ** (-poles[pole]))
        else:
            if im(pole) > 0:
                Q = (s - pole) * (s - pole.conjugate())
                q0 = (Q.subs(s, 0)).simplify()
                q1 = (Q.simplify()).coeff(s)
                H /= (s**2 / q0 + q1 / q0 * s + 1) ** poles[pole]
                C /= q0 ** poles[pole]
                Qs.append(((s**2 / q0 + q1 / q0 * s + 1) ** (-poles[pole])))

    H *= C

    fig, ax = plt.subplots(2, 1, sharex=True)
    w0 = np.logspace(*plot_range, 1000)
    ax[0].semilogx(w0, lambdify(w, gain(H.subs(s, jw)))(w0), color="black", label="H")
    ax[1].semilogx(w0, lambdify(w, phase(H.subs(s, jw)))(w0), color="black", label="H")
    for L in Ls:
        Lw = L.subs(s, jw)
        gainL = lambdify(w, gain(Lw))
        phaseL = lambdify(w, phase(Lw))
        ax[0].semilogx(
            w0,
            gainL(w0),
            label=r"$L_{{{index}}}^{{{power}}}$".format(
                index=Ls.index(L) + 1, power=Lw.as_base_exp()[1]
            ),
            linestyle="--",
        )
        ax[1].semilogx(w0, phaseL(w0), linestyle="--")

    for Q in Qs:
        Qw = Q.subs(s, jw)
        gainQ = lambdify(w, gain(Qw))
        phaseQ = lambdify(w, phase(Qw))
        ax[0].semilogx(
            w0,
            gainQ(w0),
            label=r"$Q_{{{index}}}^{{{power}}}$".format(
                index=Qs.index(Q) + 1, power=Q.as_base_exp()[1]
            ),
            linestyle="--",
        )
        ax[1].semilogx(w0, phaseQ(w0), linestyle="--")

    for O in Os:
        Ow = O.subs(s, jw)
        gainO = lambdify(w, gain(Ow))
        phaseO = lambdify(w, phase(Ow))
        ax[0].semilogx(
            w0,
            gainO(w0),
            label=r"$\Omega_{{{index}}}^{{{power}}}$".format(
                index=Os.index(O) + 1, power=O.as_base_exp()[1], linestyle="--"
            ),
        )
        ax[1].semilogx(w0, phaseO(w0), linestyle="--")

    ax[0].axhline(gain(C), color="red", linestyle="-.", label="C")
    ax[1].axhline(phase(C), color="red", linestyle="-.")
    ax[0].grid(True, which="both")
    ax[1].grid(True, which="both")
    ax[0].legend()
    fig.tight_layout()
    return


def sDiff(f, s=Symbol("s"), t=Symbol("t", real=True)):
    f = collect(f, s)
    res = 0
    for i in range(degree(f, s) + 1):
        res += diff(f.coeff(s, i), t, i)
    return res


def bode(H, sym=Symbol("s"), pltrange=(-1, 3)):
    w = np.logspace(pltrange[0], pltrange[1], 1000)
    H = lambdify(sym, H)
    plt.figure()
    plt.subplot(211)
    plt.semilogx(w, 20 * np.log10(abs(H(1j * w))))
    plt.grid(which="both")
    plt.subplot(212)
    plt.semilogx(w, np.rad2deg(np.angle(H(1j * w))))
    plt.grid(which="both")
    plt.show()


def wnDamp(H, sym=Symbol("s")):
    damp, wn = symbols("damp wn", real=True)
    Q = denom(H)
    Q = Q.collect(sym).expand()
    Q = Q / Q.coeff(sym, 2)
    wn = sqrt(Q.subs(sym, 0))
    damp = Q.coeff(sym, 1) / (2 * wn)
    return wn, damp


def pole(H, sym=Symbol("s")):
    Q = denom(H)
    return roots(Q, sym)


def zeros(H, sym=Symbol("s")):
    P = numer(H)
    return roots(P, sym)


def ROC(H, sym=Symbol("s")):
    Q = denom(H)
    sol = solve(Q, sym, dict=False)
    sol = [re(sol0) for sol0 in sol]
    return max(sol)


def step(H, s=Symbol("s"), t=Symbol("t", real=True)):
    res = inverse_laplace_transform(H / s, s, t)
    return res.collect(Heaviside(t))


def impulse(H, s=Symbol("s"), t=Symbol("t", real=True)):
    res = inverse_laplace_transform(H, s, t)
    return res.collect(Heaviside(t))


def ramp(H, s=Symbol("s"), t=Symbol("t", real=True)):
    res = inverse_laplace_transform(H / s**2, s, t)
    return res.collect(Heaviside(t))


def odeImpulse(H, ics_0neg, s=Symbol("s"), t=Symbol("t", real=True), y="y", x="x"):
    x, y = symbols(x + " " + y, cls=Function)
    H = cancel(H, s)
    P = numer(H)
    Q = denom(H)

    if y(t).diff(t).subs(t, 0) in ics_0neg.keys():
        ics_0neg[y(t).diff(t).subs(t, 0)] = ics_0neg[y(t).diff(t).subs(t, 0)] + 1
    else:
        ics_0neg[y(t).diff(t).subs(t, 0)] = 1
    Qy = (Q * y(t)).expand()
    Qy = sDiff(Qy)
    yn = dsolve(Qy, y(t), ics=ics_0neg)
    h = sDiff(yn.rhs * P) * Heaviside(t) + P.coeff(s, degree(Q, s)) * DiracDelta(t)
    return h


def convolve(f1, f2, t="t"):
    tau = Symbol("tau", real=True) if t != "tau" else Symbol("tau_0", real=True)
    t = Symbol(t, real=True)
    res = integrate(f1.subs(t, t - tau) * f2.subs(t, tau), (tau, -oo, t))
    return res


def fixH(f, t="t"):
    t = Symbol(t, real=True)
    u = Heaviside(t)
    f = f.collect(u)
    f = f / (u ** (degree(f, u) - 1))
    return f


def laplaceI(ode, y="y", t="t", s="s"):
    y = Function(y)
    t = Symbol(t, real=True)
    s = Symbol(s)
    res = 0
    ord = ode_order(ode, y)
    for i in range(ord + 1):
        term = 0
        k = ode.coeff(y(t).diff(t, i))
        for j in range(i):
            term += y(t).diff(t, j).subs(t, 0) * s ** (i - 1 - j)
        res += k * term
    return res


def ode2tf(ode, y="y", x="x", t="t", s="s"):
    y = Function(y)
    x = Function(x)
    t = Symbol(t, real=True)
    s = Symbol(s)
    Q = 0
    P = 0
    ord = ode_order(ode.lhs, y)
    for i in range(ord + 1):
        Q += ode.lhs.coeff(y(t).diff(t, i)) * s**i

    ord = ode_order(ode.rhs, x)
    for i in range(ord + 1):
        P += ode.rhs.coeff(x(t).diff(t, i)) * s**i
    return cancel(P / Q)


def lap_with_ic(ode, y="y", x="x", t="t", s="s", ic={}, tf_out=1):

    H = ode2tf(ode, y, x, t, s)
    zi = laplaceI(ode.lhs, y, t, s)
    zi = zi.subs(ic)
    return H * tf_out + zi / denom(H)


def sensitivity(y, x):
    return diff(y, x) * x / y


def settling_time(H, s="s"):
    s = Symbol(s)
    wn, damp = wnDamp(H, s)
    return 4 / (damp * wn)


def peak_time(H, s="s"):
    s = Symbol(s)
    wn, damp = wnDamp(H, s)
    return pi / (wn * sqrt(1 - damp**2))


def overshoot(H, s="s"):
    s = Symbol(s)
    wn, damp = wnDamp(H, s)
    return 100 * exp(-pi * damp / sqrt(1 - damp**2))


def rise_time(H, s="s"):
    s = Symbol(s)
    wn, damp = wnDamp(H, s)
    return (1 - 0.4167 * damp + 2.917 * damp**2) / wn


def delay_time(H, s="s"):
    s = Symbol(s)
    wn, damp = wnDamp(H, s)
    return (1.1 + 0.125 * damp + 0.469 * damp**2) / wn


def step_info(H, s="s"):
    return {
        "Overshoot": overshoot(H, s).evalf(),
        "Peak Time": peak_time(H, s).evalf(),
        "Settling Time": settling_time(H, s).evalf(),
        "Rise Time": rise_time(H, s).evalf(),
        "Delay Time": delay_time(H, s).evalf(),
    }
