# HYPPE: how the model works

A promoter is 800 letters. Somewhere in them is the answer to one question:
how strongly does this switch turn a gene on?

HYPPE reads that answer. It also writes it back. The model turns a promoter
into a short list of discrete symbols. Those symbols can be edited, and
decoded into a new promoter.

This page explains the model twice. Each section starts with the idea in plain
words. Then it gives the mathematics. The last sections cover how we chose the
ten synthetic promoters for the wet lab, and what our numbers do not claim.

---

## 1. The shape of the problem

Three things make promoter strength hard to learn. Each one shaped a design
decision.

**The label is noisy and borrowed.** Other laboratories measured the
expression. Their studies were never meant to be compared with each other. We
pool them and correct the between-study component. What is left still carries
the spread of the original experiments. So the model predicts a mean *and* its
own uncertainty. And we judge it on ordering first, on level second.

**The examples are not independent.** One gene has orthologues in twenty
related strains. They share most of their promoter. Split them at random and
the model is tested on near-copies of its own training data. The score looks
good. It does not survive on a new locus. So our split groups whole families.

**A dense vector cannot be edited.** Suppose a model predicts well but stores
a promoter as 512 anonymous floating-point numbers. It tells a biologist
nothing about *where* the strength lives. It offers no handle for designing a
new sequence. So our representation is discrete and hierarchical. Coarse
symbols hold broad position. Finer symbols hold detail. Each one is tied to a
length in base pairs.

### What we assume, and why

Five assumptions carry the design. Each is stated here with the reason we think
it is safe, and each is tested somewhere later in this page.

| assumption | why we make it | where it is tested |
|---|---|---|
| Expression pooled from many studies is comparable once the between-study component is removed | the labels are borrowed; nobody measured this corpus for us | section 6, and we judge on ordering before level |
| A promoter's codes must not depend on the growth condition | otherwise editing the codes is not a statement about the sequence, and the design loop has nothing to steer | enforced structurally, Figure 1, path 1 |
| The hierarchy we quantise against is a tree because we imposed one, not because the data is tree-shaped | we measured tree-likeness on regulatory sequence and it is no more tree-like than random points | section 3, stated against our own interest |
| Orthologues of one gene are not independent examples | they share most of their promoter, so a random split tests the model on near-copies of its training data | section 6 |
| The phylogeny is a prior on the strain space, not a constraint | 439 protein markers place the strains, but expression may disagree and should win | section 3; section 10.2 is the case where it did not win and we loosened it |

**On measurements: the wet lab is the test, not an input.** No measurement from
the target strain P1 entered training, and no measurement of any kind that we
made ourselves did either: the corpus is public RNA-seq from other
laboratories. The strain enters by its learned position on the phylogeny, the
condition by its written description, and the plasmid context by a learned
"unknown" token. We say this plainly because it cuts both ways: it makes the
ten designs a genuine zero-shot prediction, and it means the model has never
seen the assay it is about to be judged against. Section 9 lists the three
domain shifts that sit between our label and a wet read-out.

---

## 2. The architecture at a glance

<figure>
<svg viewBox="0 0 900 520" role="img" aria-label="The HYPPE architecture, laid out as the letter H. The left upright carries the inputs and the Encoder: a promoter and its motif occupancy both feed the Encoder, which sits at the junction. The crossbar is the Cone Ladder, the hyperbolic quantiser the model is named for. The right upright carries the Codes at its junction, with the Decoder above and the prediction Heads below, and the Condition fused into the heads. Two paths are crossed out in red: the Condition never reaches the Encoder, and no continuous path reaches the Decoder." style="max-width:100%;height:auto;color:#15202b;font-family:system-ui,-apple-system,Segoe UI,sans-serif">
  <defs>
    <marker id="ha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
    <marker id="hb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#d66a1a"/>
    </marker>
  </defs>

  <!-- the two uprights and the crossbar of the H -->
  <g fill="none" stroke="currentColor" stroke-width="2">
    <rect x="50" y="30" width="210" height="70" rx="6"/>
    <rect x="50" y="180" width="210" height="140" rx="6"/>
    <rect x="50" y="400" width="210" height="70" rx="6"/>
    <rect x="630" y="30" width="220" height="100" rx="6"/>
    <rect x="630" y="180" width="220" height="140" rx="6"/>
    <rect x="630" y="380" width="220" height="120" rx="6"/>
  </g>
  <rect x="290" y="180" width="310" height="140" rx="6" fill="none" stroke="#d66a1a" stroke-width="3"/>
  <rect x="320" y="350" width="280" height="68" rx="6" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="6 4"/>

  <g text-anchor="middle">
    <text x="155" y="72" font-size="18" font-weight="600">Promoter</text>

    <text x="155" y="245" font-size="18" font-weight="600">Encoder</text>
    <text x="155" y="272" font-size="14" fill="#555">Five scales</text>

    <text x="155" y="441" font-size="18" font-weight="600">Motifs</text>

    <text x="445" y="236" font-size="20" font-weight="600" fill="#d66a1a">Cone Ladder</text>
    <text x="445" y="270" font-size="15" fill="#d66a1a">L0 position, L1–L4 detail</text>

    <text x="740" y="76" font-size="18" font-weight="600">Decoder</text>

    <text x="740" y="245" font-size="18" font-weight="600">Codes</text>
    <text x="740" y="272" font-size="14" fill="#555">Five levels</text>

    <text x="740" y="428" font-size="18" font-weight="600">Heads</text>
    <text x="740" y="456" font-size="14" fill="#555">Four predictions</text>

    <text x="460" y="382" font-size="17" font-weight="600">Condition</text>
    <text x="460" y="406" font-size="14" fill="#555">Strain, description, flanks</text>
  </g>

  <!-- flow along the uprights and across the bar -->
  <g stroke="currentColor" stroke-width="2" fill="none" marker-end="url(#ha)">
    <path d="M155 100 L155 176"/>
    <path d="M155 400 L155 324"/>
    <path d="M740 180 L740 134"/>
    <path d="M740 320 L740 376"/>
    <path d="M600 384 L626 384"/>
  </g>
  <g stroke="#d66a1a" stroke-width="2.4" fill="none" marker-end="url(#hb)">
    <path d="M260 250 L286 250"/>
    <path d="M600 250 L626 250"/>
  </g>

  <!-- the two constraints live in the hollows of the H -->
  <g stroke="#b3261e" stroke-width="2" fill="none" stroke-dasharray="6 4">
    <path d="M320 384 L190 384 L190 324"/>
    <path d="M260 200 L260 115 L624 115"/>
  </g>
  <g stroke="#b3261e" stroke-width="3">
    <path d="M243 372 L267 396 M267 372 L243 396"/>
    <path d="M433 103 L457 127 M457 103 L433 127"/>
  </g>
  <!-- Callouts sit ON the dashed path but AWAY from the crosses, in a filled
       disc that masks the line behind them. Set as bare numerals they
       collided with the cross strokes and, for 2, the dashed rule ran
       straight through the glyph. -->
  <g>
    <circle cx="190" cy="352" r="12" fill="#ffffff" stroke="#b3261e" stroke-width="2"/>
    <text x="190" y="357" text-anchor="middle" font-size="14" font-weight="700" fill="#b3261e">1</text>
    <circle cx="540" cy="115" r="12" fill="#ffffff" stroke="#b3261e" stroke-width="2"/>
    <text x="540" y="120" text-anchor="middle" font-size="14" font-weight="700" fill="#b3261e">2</text>
  </g>

</svg>
<figcaption><b>Figure 1.</b> The whole model. A promoter and its motif
occupancy go into the Encoder, the Cone Ladder turns that into discrete Codes,
and the Codes feed both a Decoder that rebuilds the sequence and the Heads that
predict how strongly it fires. The two crossed paths are the choices that make
the Codes usable for design, and the code enforces both: the growth Condition
never reaches the Encoder, so a promoter always gets the same Codes whatever
the experiment; and nothing but the Codes reaches the Decoder, so anything they
fail to carry is visibly lost rather than quietly smuggled
through.</figcaption>
</figure>

**Where each piece comes from.** The hyperboloid and its optimisation follow
Nickel & Kiela (2018) and Bécigneul & Ganea (2019); the level-0 codebook is a
VQ-VAE codebook in the sense of van den Oord et al. (2017) and the finer levels
use the finite scalar quantisation of Mentzer et al. (2024). The Decoder is
held to its target by the constrained-optimisation scheme of Rezende & Viola
(2018) rather than by a hand-set loss weight. The Strength head uses the
beta-NLL objective of Seitzer et al. (2022), the Ranking head a
Bradley & Terry (1952) likelihood, and the Condition's description channel the
sentence embedding of Wang et al. (2020). Full references are at the end.

Those two crossed paths are the architecture's load-bearing choices.

**The decoder sees only codes.** Suppose the decoder could also read the
encoder's continuous features. It would use them and ignore the codes, and the
codes would become decoration. We measured exactly that on a sibling model.
With a continuous bypass open, reconstruction read 0.9984. Reconstruction from
the codes alone read 0.29, against a chance floor of 0.26. Closing the bypass
is what makes "decode the codes into a sequence" a real operation.

**The condition never reaches the encoder.** The strain, the growth condition
and the neighbouring DNA all matter for *how strongly* a promoter fires. But
they are not part of the promoter. We keep them behind the quantiser. The code
of a sequence is then a function of the sequence alone. The same promoter has
the same codes in every strain, and the condition moves only the prediction.

---

## 3. Why hyperbolic, honestly

Hyperbolic space is the natural home of trees. The number of points at
distance *r* from a centre grows exponentially with *r*. So does the number of
nodes at depth *r* in a branching tree. Euclidean space grows polynomially, so
a deep tree embedded in it crowds the leaves.

The usual argument for using it on biological sequence is that the data are
hierarchical. **That argument does not survive measurement, and we checked.**
We used the standard measure of tree-likeness. We applied it to a
regulatory-sequence corpus and to a motif bank in the sibling track of this
codebase, not to the *Trichoderma* corpus. Regulatory sequence is not
meaningfully more tree-like than random points. The motif space came out
*less* tree-like than random Euclidean points.

One thing here is a tree by construction: the code hierarchy we impose. One
coarse symbol, and under it the refinements that specialise it. That, and only
that, is why the geometry is hyperbolic. We say it this way because the
opposite claim is easy to make and would be false.

### The hyperboloid

We work on the upper sheet of a two-sheeted hyperboloid. Take the Minkowski
form on $\mathbb{R}^{n+1}$:

$$\langle x, y\rangle_{\mathcal L} \;=\; -x_0y_0 \;+\; \sum_{i=1}^{n} x_i y_i,$$

The model space is

$$\mathbb{H}^n \;=\; \bigl\{\, x \in \mathbb{R}^{n+1} \;:\; \langle x,x\rangle_{\mathcal L} = -1,\; x_0 > 0 \,\bigr\},$$

and the distance between two of its points is

$$d(x,y) \;=\; \operatorname{arcosh}\!\bigl(-\langle x,y\rangle_{\mathcal L}\bigr).$$

Three operations do all the work. The first moves from a point $p$ along a
geodesic, with initial velocity $v$ in the tangent space $T_p$:

$$\exp_p(v) \;=\; \cosh\!\bigl(\lVert v\rVert_{\mathcal L}\bigr)\, p \;+\; \sinh\!\bigl(\lVert v\rVert_{\mathcal L}\bigr)\,\frac{v}{\lVert v\rVert_{\mathcal L}},$$

The second is its inverse $\log_p(q)$. It returns the velocity that takes you
from $p$ to $q$. The third is parallel transport $\mathrm{PT}_{p\to q}(v)$. It
carries a tangent vector from one point to another without rotating it.

<figure>
<svg viewBox="100 140 450 185" role="img" aria-label="A cross-section of the upper sheet of the hyperboloid, drawn as a curve opening upward between its two dashed asymptotes. The origin sits at the bottom of the curve. The flat tangent space at the origin is drawn as a horizontal line carrying a straight arrow v. The exponential map wraps that straight arrow onto the surface: the image is a geodesic of the same length running along the curve, ending at the point p equals exp of v." style="max-width:100%;height:auto;color:#1b1b1b;font-family:system-ui,-apple-system,Segoe UI,sans-serif">
  <defs>
    <marker id="ma" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#d66a1a"/>
    </marker>
    <marker id="mg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#8c3e0a"/>
    </marker>
  </defs>

  <!-- asymptotes: the light cone the sheet never reaches -->
  <g stroke="#bbb" stroke-width="1" stroke-dasharray="5 5">
    <line x1="290" y1="312" x2="468" y2="158"/>
    <line x1="290" y1="312" x2="112" y2="158"/>
  </g>

  <!-- the sheet, in cross-section -->
  <path d="M119.0 152.2 L181.3 199.7 L226.1 229.1 L260.4 245.0 L290 250
           L319.6 245.0 L353.9 229.1 L398.7 199.7 L461.0 152.2"
        fill="none" stroke="#8c3e0a" stroke-width="2.2"/>

  <!-- the tangent space at the origin is FLAT: that is the whole point -->
  <line x1="196" y1="250" x2="420" y2="250" stroke="#999" stroke-width="1.2"/>
  <text x="428" y="255" font-size="14" fill="#777">Tangent space</text>

  <!-- v, straight, in the tangent space -->
  <line x1="290" y1="250" x2="371" y2="250" stroke="#d66a1a" stroke-width="2.6" marker-end="url(#ma)"/>
  <text x="330" y="274" text-anchor="middle" font-size="15" font-weight="600" fill="#d66a1a">v</text>

  <!-- its image: a geodesic ALONG the sheet, of the same length -->
  <path d="M290 250 L308.2 248.1 L327.5 242.1 L349.2 231.7 L374.6 216.3"
        fill="none" stroke="#d66a1a" stroke-width="4.2" stroke-linecap="round" opacity="0.85"/>

  <!-- exp wraps the straight arrow onto the surface, log unwraps it -->
  <path d="M374 246 A 46 46 0 0 0 377 223" fill="none" stroke="#d66a1a"
        stroke-width="1.6" marker-end="url(#ma)"/>

  <circle cx="290" cy="250" r="5" fill="#1b1b1b"/>
  <text x="290" y="294" text-anchor="middle" font-size="14">Origin</text>
  <line x1="290" y1="258" x2="290" y2="280" stroke="#999" stroke-width="0.8" stroke-dasharray="2 3"/>

  <circle cx="374.6" cy="216.3" r="6" fill="#d66a1a"/>
  <text x="392" y="212" font-size="15" font-weight="600" fill="#d66a1a">p</text>

  <text x="300" y="194" text-anchor="middle" font-size="14" fill="#8c3e0a">Geodesic</text>
  <line x1="316" y1="201" x2="334" y2="226" stroke="#999" stroke-width="0.9" marker-end="url(#mg)"/>
</svg>
<figcaption><b>Figure 2.</b> How a flat vector becomes a point on a curved
surface. The network produces an ordinary straight vector <i>v</i>; the
exponential map wraps it onto the hyperboloid as a curve of the same length,
ending at <i>p</i>. This is the only bridge between the flat space the network
computes in and the curved space the Codes live in, and every rung of the Cone
Ladder is one use of it.</figcaption>
</figure>

Transport is not a refinement of the method. A vector that is tangent at the
origin is *not* tangent somewhere else. Transport is what makes the
composition in the next section type-correct at all.

### The one place the geometry holds data rather than codes

Two things in this model live on a hyperboloid, and only one of them has a real
tree behind it.

The Cone Ladder is hyperbolic because we built a hierarchy into it: coarse
symbols with finer ones underneath. That tree is our design decision, not
something we found in the data, which is exactly the caveat above.

The **strain registry** is the other one, and its tree is not ours. The twenty
genomes are real organisms with a real evolutionary history, and we have that
history as a phylogeny built from 439 protein markers. So each genome gets a
point on a three-dimensional hyperboloid, and a penalty term pulls the
distances between those points toward the distances along the branches of that
tree. The penalty is soft: it is just added to the training loss, so the
expression data can push back against it. Section 10.2 describes the case where
it pushed back too weakly and we had to loosen the penalty by hand.

Figure 3 is that registry, read straight out of the delivered checkpoint.

<figure>
<svg viewBox="0 0 880 470" role="img" aria-label="A dot plot of twenty genomes ordered by geodesic distance from strain P1 in the learned strain embedding. The ten Trichoderma atroviride strains, P1 itself, F7, B10, I1237, P3116, P3041, P3080, IMI206040, N1508 and MMS1295, all fall within 0.167 of P1 and form a shaded cluster at the left edge. Every other species lies between 0.472 for viride and 2.370 for reesei, so the nearest other species is 2.8 times further away than the furthest strain of the same species." style="max-width:100%;height:auto;color:#1b1b1b;font-family:system-ui,-apple-system,Segoe UI,sans-serif">
  <text x="8" y="20" font-size="15" font-weight="600">Distance from P1 in the learned strain space</text>

  <!-- the within-species band -->
  <rect x="170" y="48" width="45.3" height="172" fill="#fceee0"/>
  <text x="222" y="40" font-size="14" fill="#8c3e0a">Same species as our target</text>
  <text x="872" y="40" text-anchor="end" font-size="14" fill="#8c3e0a">Other species</text>

  <g font-size="13" text-anchor="end">
    <text x="152" y="62" font-weight="700">P1</text>
    <text x="152" y="79">F7</text>
    <text x="152" y="96">B10</text>
    <text x="152" y="113">I1237</text>
    <text x="152" y="130">P3116</text>
    <text x="152" y="147">P3041</text>
    <text x="152" y="164">P3080</text>
    <text x="152" y="181">IMI206040</text>
    <text x="152" y="198">N1508</text>
    <text x="152" y="215">MMS1295</text>
    <text x="152" y="232" font-style="italic">viride</text>
    <text x="152" y="249" font-style="italic">asperellum</text>
    <text x="152" y="266" font-style="italic">asperelloides</text>
    <text x="152" y="283" font-style="italic">virens</text>
    <text x="152" y="300" font-style="italic">guizhouense</text>
    <text x="152" y="317" font-style="italic">harzianum</text>
    <text x="152" y="334" font-style="italic">velutinum</text>
    <text x="152" y="351" font-style="italic">citrinoviride</text>
    <text x="152" y="368" font-style="italic">longibrachiatum</text>
    <text x="152" y="385" font-style="italic">reesei</text>
  </g>

  <!-- leaders -->
  <g stroke="#bbb" stroke-width="0.7" stroke-dasharray="2 3">
    <line x1="158" y1="58" x2="164" y2="58"/>
    <line x1="158" y1="75" x2="175" y2="75"/>
    <line x1="158" y1="92" x2="175" y2="92"/>
    <line x1="158" y1="109" x2="180" y2="109"/>
    <line x1="158" y1="126" x2="180" y2="126"/>
    <line x1="158" y1="143" x2="180" y2="143"/>
    <line x1="158" y1="160" x2="180" y2="160"/>
    <line x1="158" y1="177" x2="180" y2="177"/>
    <line x1="158" y1="194" x2="209" y2="194"/>
    <line x1="158" y1="211" x2="209" y2="211"/>
    <line x1="158" y1="228" x2="292" y2="228"/>
    <line x1="158" y1="245" x2="387" y2="245"/>
    <line x1="158" y1="262" x2="393" y2="262"/>
    <line x1="158" y1="279" x2="694" y2="279"/>
    <line x1="158" y1="296" x2="707" y2="296"/>
    <line x1="158" y1="313" x2="715" y2="313"/>
    <line x1="158" y1="330" x2="732" y2="330"/>
    <line x1="158" y1="347" x2="782" y2="347"/>
    <line x1="158" y1="364" x2="786" y2="364"/>
    <line x1="158" y1="381" x2="806" y2="381"/>
  </g>

  <!-- same species: orange.  other species: dark -->
  <g fill="#d66a1a">
    <circle cx="170.0" cy="58" r="5.5"/>
    <circle cx="181.0" cy="75" r="5"/>
    <circle cx="181.1" cy="92" r="5"/>
    <circle cx="185.7" cy="109" r="5"/>
    <circle cx="185.7" cy="126" r="5"/>
    <circle cx="185.7" cy="143" r="5"/>
    <circle cx="185.7" cy="160" r="5"/>
    <circle cx="186.0" cy="177" r="5"/>
    <circle cx="215.3" cy="194" r="5"/>
    <circle cx="215.3" cy="211" r="5"/>
  </g>
  <g fill="#8c3e0a">
    <circle cx="297.9" cy="228" r="5"/>
    <circle cx="393.5" cy="245" r="5"/>
    <circle cx="399.4" cy="262" r="5"/>
    <circle cx="699.8" cy="279" r="5"/>
    <circle cx="713.2" cy="296" r="5"/>
    <circle cx="720.6" cy="313" r="5"/>
    <circle cx="738.2" cy="330" r="5"/>
    <circle cx="788.0" cy="347" r="5"/>
    <circle cx="791.6" cy="364" r="5"/>
    <circle cx="811.8" cy="381" r="5"/>
  </g>

  <!-- axis -->
  <line x1="170" y1="402" x2="850" y2="402" stroke="#1b1b1b" stroke-width="1.1"/>
  <g stroke="#1b1b1b" stroke-width="1.1">
    <path d="M170 402v6 M305.4 402v6 M440.8 402v6 M576.2 402v6 M711.7 402v6 M847.1 402v6"/>
  </g>
  <g font-size="13" text-anchor="middle" fill="#444">
    <text x="170" y="424">0</text>
    <text x="305.4" y="424">0.5</text>
    <text x="440.8" y="424">1.0</text>
    <text x="576.2" y="424">1.5</text>
    <text x="711.7" y="424">2.0</text>
    <text x="847.1" y="424">2.5</text>
  </g>
  <text x="508" y="448" text-anchor="middle" font-size="15">Distance on the hyperboloid</text>
</svg>
<figcaption><b>Figure 3.</b> Where the model puts each of the twenty genomes,
read straight out of the delivered checkpoint. The ten <i>T. atroviride</i>
strains, our design target P1 among them, all sit within 0.167 of each other,
while the nearest other species is at 0.472 and the furthest at 2.370. So the
model has clearly learned the species boundary, and has learned very little
below it. That is the shape a relational embedding is meant to have, and it is
also why a plain lookup table beats it at telling apart strains it has already
seen.</figcaption>
</figure>

Two readings, and the second is the uncomfortable one. The ten *T.
atroviride* strains, our design target P1 among them, fall within **0.167** of
each other, while the nearest other species sits at **0.472** and the furthest
at **2.370**, so crossing the species boundary costs at least 2.8× the widest
gap inside the species. The embedding has learned the taxonomy.

It has not learned much below it. Six of these strains share the `pks1`
promoter byte for byte while their measured expression spans 0.039 to 7.624,
and a geometry that puts them 0.06 apart cannot be what separates them. We
measured what the alternative is worth by holding each strain out in turn and
refitting: a flat one-hot gains **+0.032 on a strain it was fitted on and
−0.007 on a strain it never saw** ([ablations.md](ablations.md), seventeen
strains). Its whole value is memorised separation, and P1 is a strain the
pipeline is zero-shot against, so we keep the embedding and report the loss
it costs us on the strains we did see.

---

## 4. The cone ladder

Here is the central idea in one sentence: **level 0 says where the promoter
is, and every later level says how to adjust that.** The adjustments come from
a finite menu.

<figure>
<svg viewBox="0 0 700 385" role="img" aria-label="A cross-section of the hyperboloid, drawn as three dashed shells at radius r, r plus s and r plus 2s around the origin. The coarse level places p0 on the inner shell. Each refinement is a short bounded tangent step that also pushes the point one fixed increment further out, so p1 lands on the next shell and p2 on the one after. Depth in the ladder is radius in the space." style="max-width:100%;height:auto;color:#1b1b1b;font-family:system-ui,-apple-system,Segoe UI,sans-serif">
  <defs>
    <marker id="a2b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#d66a1a"/>
    </marker>
  </defs>

  <!-- three shells of equal radial spacing s -->
  <g fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="3 4">
    <path d="M230.96 288.70 A150 150 0 0 0 128.82 195.11"/>
    <path d="M296.73 264.76 A220 220 0 0 0 146.94 127.50"/>
    <path d="M357.81 242.52 A285 285 0 0 0 163.76 64.72"/>
  </g>
  <g font-size="13" fill="#555">
    <text x="238" y="303">r</text>
    <text x="304" y="279">r + s</text>
    <text x="365" y="257">r + 2s</text>
  </g>

  <circle cx="90" cy="340" r="4.5" fill="currentColor"/>
  <text x="90" y="363" text-anchor="middle" font-size="14">Origin</text>
  <line x1="90" y1="340" x2="182" y2="222" stroke="currentColor" stroke-width="1.1"/>

  <!-- the ladder: one bounded tangent step per level, each landing a shell out -->
  <line x1="188" y1="214" x2="204" y2="163" stroke="#d66a1a" stroke-width="2.2" marker-end="url(#a2b)"/>
  <line x1="212" y1="145" x2="216" y2="95" stroke="#d66a1a" stroke-width="2.2" marker-end="url(#a2b)"/>
  <text x="178" y="190" text-anchor="end" font-size="15" font-weight="600" fill="#d66a1a">v</text>

  <circle cx="182.4" cy="221.8" r="6" fill="#d66a1a"/>
  <circle cx="206.6" cy="153.4" r="6" fill="#d66a1a"/>
  <circle cx="214.9" cy="83.8" r="6" fill="#d66a1a"/>

  <!-- labels live in a column clear of every shell, on dotted leaders -->
  <g stroke="#999" stroke-width="0.8" stroke-dasharray="2 3">
    <line x1="190" y1="221.8" x2="372" y2="221.8"/>
    <line x1="214" y1="153.4" x2="372" y2="153.4"/>
    <line x1="222" y1="83.8" x2="372" y2="83.8"/>
  </g>
  <g font-size="17" font-weight="600" fill="#d66a1a">
    <text x="380" y="227">p₀</text>
    <text x="380" y="159">p₁</text>
    <text x="380" y="89">p₂</text>
  </g>

</svg>
<figcaption><b>Figure 4.</b> How the ladder is built. The coarse code places
p₀ on the inner shell: it is one of 128 learned points. Each finer level adds
a short, bounded step from its parent: p₁, then p₂, and so on down to L4. Every
step also pushes the point one fixed increment further out, so depth in the
hierarchy is literally radius in the space. Without that push the radius
wanders, all five levels
settle onto a single shell, and "coarse" and "fine" stop meaning anything.</figcaption>
</figure>

Write $c_k$ for the code chosen at level $k$:

$$p_0 \;=\; \mathrm{codebook}[c_0], \qquad p_{k+1} \;=\; \exp_{p_k}\!\Bigl(\mathrm{PT}_{o\to p_k}\bigl(v_{k+1}\bigr) \;+\; s_k\,\mathrm{radial}(p_k)\Bigr),$$

Here $o$ is the origin. The vector $v_{k+1}$ is the quantised tangent vector of
level $k+1$. The step $s_k = 0.15$ is the fixed radial increment. And
$\mathrm{radial}(p)$ is the unit tangent at $p$ pointing straight away from the
origin.

Two properties follow. Both are used downstream.

**Containment.** An aperture caps the tangent step, so a child can only be so
far from its parent. Coarse codes therefore constrain fine ones. That is what
"hierarchy" has to mean in practice.

**The prefix property.** Every refinement grid contains *exactly zero*. A
level can say "no change". So the ladder can be truncated after any level and
still give a legal code that decodes to a sequence. That is why we can ask
what a promoter looks like with only its coarse structure kept, and get an
answer rather than an error.

### Quantisation

Level 0 picks from a learned codebook of 128 points **on the manifold**. It
takes the nearest one by geodesic distance.

Levels 1 to 4 use finite scalar quantisation. The encoder's feature is
squashed to $[-1,1]$, rounded to a grid of three values per axis, and read out
through a fixed orthonormal frame:

$$z = W x,\qquad y = \tanh(z)\cdot \tfrac{L-1}{2},\qquad \hat y = \operatorname{round}(y),$$

Gradients pass through the rounding by the straight-through estimator,
$\hat y + (y - \mathrm{sg}[y])$. The forward value is exact, and the backward
pass behaves as if rounding were the identity. An odd number of levels per
axis is what guarantees the exact zero that the prefix property needs.

An exponential moving average maintains the codebook. Its target is the
**Fréchet mean**, the point that minimises the sum of squared geodesic
distances. It is not the arithmetic mean of coordinates followed by a
projection. On the outer shells the two differ by more than two units of
distance, and the naive version silently drags codes off the manifold.

### The five levels, as they are set in the delivered model

Read out of the shipped checkpoint rather than from the config, because the
two can disagree and only one of them is the model:

| level | quantiser | alphabet | grid axes | aperture $\lVert v\rVert$ | shell radius |
|---|---|---|---|---|---|
| L0 | geodesic codebook on $\mathbb{H}^{16}$ | 128 | — | — | 0.924 to 1.940 |
| L1 | FSQ, 3 values per axis | 729 | 6 | 0.542 | $\bar r + 0.15$ |
| L2 | FSQ | 81 | 4 | 0.666 | $\bar r + 0.30$ |
| L3 | FSQ | 9 | 2 | 0.869 | $\bar r + 0.45$ |
| L4 | FSQ | 9 | 2 | 1.130 | $\bar r + 0.60$ |

The radial step is $s = 0.15$ at every level and the mean L0 radius is
$\bar r = 1.626$.

**The aperture schedule is not arbitrary, and what it holds constant is
angular.** On a shell of radius $r$ a geodesic step of length $d$ subtends an
angle of about $d/\sinh r$, and $\sinh r$ grows with depth because every level
is pushed one step further out. Measured on this checkpoint, the aperture
grows **2.08×** from L1 to L4 while the angle it buys grows only **1.31×**.
Held at L1's aperture instead, the angular reach would *fall by 37%* by L4,
and the deepest levels, the ones that write individual bases, would be the
most tightly cornered. The schedule pays for the circumference.

### The bit budget, and the honest reading of it

| level | cells | bp per cell | alphabet | bits/cell | bits |
|---|---|---|---|---|---|
| L0 | 25 | 32 | 128 | 7.0000 | 175.0 |
| L1 | 50 | 16 | 729 | 9.5098 | 475.5 |
| L2 | 100 | 8 | 81 | 6.3399 | 634.0 |
| L3 | 200 | 4 | 9 | 3.1699 | 634.0 |
| L4 | 400 | 2 | 9 | 3.1699 | 1268.0 |
| | | | | **total** | **3186.5** |

<figure>
<svg viewBox="0 0 860 230" role="img" aria-label="Five horizontal bars over the same 800 base pairs, divided into 25, 50, 100, 200 and 400 cells, so each level corresponds to a length scale from 32 base pairs down to 2." style="max-width:100%;height:auto;color:#1b1b1b">
  <text x="12" y="20" font-family="system-ui,sans-serif" font-size="15" font-weight="600">The same 800 bp, five times</text>
  <g font-family="system-ui,sans-serif" font-size="13">
    <text x="12" y="52">L0</text><text x="44" y="52">32 bp</text>
    <text x="12" y="86">L1</text><text x="44" y="86">16 bp</text>
    <text x="12" y="120">L2</text><text x="44" y="120">8 bp</text>
    <text x="12" y="154">L3</text><text x="44" y="154">4 bp</text>
    <text x="12" y="188">L4</text><text x="44" y="188">2 bp</text>
    <text x="760" y="52">25</text>
    <text x="760" y="86">50</text>
    <text x="760" y="120">100</text>
    <text x="760" y="154">200</text>
    <text x="760" y="188">400</text>
    <text x="760" y="22">cells</text>
  </g>
  <g stroke="#d66a1a" fill="none" stroke-width="1">
    <rect x="104" y="38" width="640" height="18"/>
    <rect x="104" y="72" width="640" height="18"/>
    <rect x="104" y="106" width="640" height="18"/>
    <rect x="104" y="140" width="640" height="18"/>
    <rect x="104" y="174" width="640" height="18"/>
  </g>
  <g stroke="#d66a1a" stroke-width="0.7" opacity="0.85">
    <path d="M129.6 38v18 M155.2 38v18 M180.8 38v18 M206.4 38v18 M232 38v18 M257.6 38v18 M283.2 38v18 M308.8 38v18 M334.4 38v18 M360 38v18 M385.6 38v18 M411.2 38v18 M436.8 38v18 M462.4 38v18 M488 38v18 M513.6 38v18 M539.2 38v18 M564.8 38v18 M590.4 38v18 M616 38v18 M641.6 38v18 M667.2 38v18 M692.8 38v18 M718.4 38v18"/>
    <path d="M116.8 72v18 M129.6 72v18 M142.4 72v18 M155.2 72v18 M168 72v18 M180.8 72v18 M193.6 72v18 M206.4 72v18 M219.2 72v18 M232 72v18 M244.8 72v18 M257.6 72v18 M270.4 72v18 M283.2 72v18 M296 72v18 M308.8 72v18 M321.6 72v18 M334.4 72v18 M347.2 72v18 M360 72v18 M372.8 72v18 M385.6 72v18 M398.4 72v18 M411.2 72v18 M424 72v18 M436.8 72v18 M449.6 72v18 M462.4 72v18 M475.2 72v18 M488 72v18 M500.8 72v18 M513.6 72v18 M526.4 72v18 M539.2 72v18 M552 72v18 M564.8 72v18 M577.6 72v18 M590.4 72v18 M603.2 72v18 M616 72v18 M628.8 72v18 M641.6 72v18 M654.4 72v18 M667.2 72v18 M680 72v18 M692.8 72v18 M705.6 72v18 M718.4 72v18 M731.2 72v18"/>
  </g>
  <g stroke="#d66a1a" stroke-width="0.5" opacity="0.55">
    <path d="M110.4 106v18 M116.8 106v18 M123.2 106v18 M129.6 106v18 M136.0 106v18 M142.4 106v18 M148.8 106v18 M155.2 106v18 M161.6 106v18 M168.0 106v18 M174.4 106v18 M180.8 106v18 M187.2 106v18 M193.6 106v18 M200.0 106v18 M206.4 106v18 M212.8 106v18 M219.2 106v18 M225.6 106v18 M232.0 106v18"/>
  </g>
  <g stroke="#d66a1a" stroke-width="0.45" opacity="0.45">
    <path d="M107.2 140v18 M110.4 140v18 M113.6 140v18 M116.8 140v18 M120.0 140v18 M123.2 140v18 M126.4 140v18 M129.6 140v18 M132.8 140v18 M136.0 140v18 M139.2 140v18 M142.4 140v18 M145.6 140v18 M148.8 140v18 M152.0 140v18 M155.2 140v18 M158.4 140v18 M161.6 140v18 M164.8 140v18 M168.0 140v18 M171.2 140v18 M174.4 140v18 M177.6 140v18 M180.8 140v18 M184.0 140v18 M187.2 140v18 M190.4 140v18 M193.6 140v18"/>
  </g>
  <g stroke="#d66a1a" stroke-width="0.4" opacity="0.4">
    <path d="M105.6 174v18 M107.2 174v18 M108.8 174v18 M110.4 174v18 M112.0 174v18 M113.6 174v18 M115.2 174v18 M116.8 174v18 M118.4 174v18 M120.0 174v18 M121.6 174v18 M123.2 174v18 M124.8 174v18 M126.4 174v18 M128.0 174v18 M129.6 174v18 M131.2 174v18 M132.8 174v18 M134.4 174v18 M136.0 174v18 M137.6 174v18 M139.2 174v18 M140.8 174v18 M142.4 174v18 M144.0 174v18 M145.6 174v18 M147.2 174v18 M148.8 174v18 M150.4 174v18 M152.0 174v18 M153.6 174v18 M155.2 174v18 M156.8 174v18 M158.4 174v18"/>
  </g>
  <text x="104" y="218" font-family="system-ui,sans-serif" font-size="14">5′</text>
  <text x="744" y="218" text-anchor="end" font-family="system-ui,sans-serif" font-size="14">3′</text>
</svg>
<figcaption><b>Figure 5.</b> Every level covers the whole promoter, just at a
different resolution, from one symbol per 32 bases down to one per 2. This is
why silencing a level tells you something about a length scale rather than
about an anonymous channel. The 3′ end is the base immediately before the start
codon. On the three finest levels only the first stretch is drawn, because 400
dividing lines across this width would read as a solid block.</figcaption>
</figure>

An 800 bp sequence carries about $800 \times 1.9983 \approx 1599$ bits. The
ladder spends 3186. **That is 1.99× more than the sequence contains.**

We report that ratio next to every claim about reconstruction. The two have to
be read together. At twice the budget the latent *describes* the promoter
rather than abstracting it. Reconstruction from codes alone is essentially
perfect, and that follows from the budget. It is not a separate discovery.

The right word for this representation is **auditable**. You can see exactly
which symbol owns which 32, 16, 8, 4 or 2 bases. The wrong word is
"interpretable", which would claim the hierarchy had found grammar. It has
not.

---

## 5. What the heads predict

All four heads read the same pooled summary of the ladder. For each level we
take the mean, the standard deviation and a high-quantile statistic of its
codes. Together that is 240 numbers. The condition is fused into that vector
by feature-wise modulation, before the heads see it.

### Strength, with its own error bar

The regression head emits three things: a mean $\mu$, the log-variance of its
own error, and a separate prediction of the *measured* replicate spread.

Training a heteroscedastic Gaussian the obvious way destroyed the run in three
epochs. The loop is easy to see. Lowering $\sigma$ on easy examples raises the
relative weight of hard ones. Their $\sigma$ must then grow, which lowers the
easy ones further.

$\beta$-NLL damps that loop. It detaches a power of the variance:

$$\mathcal L_i \;=\; \mathrm{sg}\!\left[\sigma_i^{2\beta}\right]\left(\frac{(y_i-\mu_i)^2}{2\sigma_i^{2}} \;+\; \tfrac{1}{2}\log \sigma_i^{2}\right),\qquad \beta = 0.5 .$$

A second exponent on the $\mu$ path is set so the gradient reaching the mean is
exactly the mean-squared-error gradient. The variance is then an output we
read. It is never a lever the model can pull to cut its own loss on hard rows.

The two uncertainties are not tied, and that is on purpose. One says how wrong
the model is. The other says how noisy the measurement was. Fusing them would
make the training weight a function of a label.

### Which of these two is stronger

The ranking head scores each promoter alone and learns on the difference. The
model is Bradley–Terry:

$$P(A \succ B) \;=\; \sigma\!\left(\frac{s_A - s_B}{T}\right), \qquad \mathcal L \;=\; \mathrm{softplus}\!\left(-\,\mathrm{sign}\cdot\frac{s_A-s_B}{T}\right).$$

**There is no margin.** That is a deletion, not a default. A hinge loss has
exactly zero gradient once a pair is ordered past its threshold. So it stops
learning while it is still wrong about the *size* of the gap. The size of the
gap is exactly what a correlation score reads.

Pairs are drawn within one (strain, condition) cell. "Which is stronger" is
only a well-posed question when the conditions match. A reserved share of each
batch holds the gene and the condition fixed, and varies the strain instead.
Without those pairs the strain channel gets no gradient from this head at all,
because within one genome the strain is constant inside every pair.

### Is this even a promoter

The discriminator separates real promoters from synthetic negatives matched on
composition. There are two kinds: dinucleotide-preserving shuffles, and
chimeras spliced from two promoters of the same genome.

The shuffle is the head's own falsifier. It preserves composition exactly, so
a GC-only baseline must sit at chance against it. The configuration refuses to
build the head without that class.

We added the chimera class after a measurement. A 5-mer linear probe could
separate positives from shuffles at 0.80 AUC. With chimeras in the pool the
same probe drops to 0.41, against a real-versus-real band of 0.43.

### Balancing three heads on one encoder

Three tasks share one encoder, and their gradients differ in scale by orders
of magnitude. We rescale each task's gradient over the shared encoder to unit
norm, multiply by an explicit prior, and renormalise:

$$\alpha_i \;=\; k\cdot\frac{\pi_i/\lVert g_i\rVert}{\sum_j \pi_j/\lVert g_j\rVert},\qquad \pi = (1.5,\;1.0,\;0.6).$$

Reconstruction is **not** a fourth player. A Lagrange multiplier holds it at a
target, updated multiplicatively,
$\lambda \leftarrow \lambda\,e^{\eta\,(\text{target} - \text{EMA})}$.

The reason is arithmetic rather than aesthetic. A bargaining solution
equalises utilities. So it hands the largest weight to the player with the
*weakest* gradient, and a nearly solved task has the weakest gradient. We
measured this on an earlier configuration. Reconstruction took 2.31 of a
budget of 3.0, while the two prediction tasks were suppressed to 0.39 and
0.31. That task was already at 0.9998.

### Optimising on a curved space

Codebook points and strain vectors are points on the hyperboloid, not free
vectors. So they get a Riemannian optimiser. The Euclidean gradient is
converted by raising the metric, which with signature $(-,+,\dots,+)$ means
flipping the time component. The update is an exponential map, not an
addition. Momentum is parallel-transported to the new point. Running a
standard optimiser on those tensors is refused, not warned about.

---

## 6. How the data were split

Twenty strains, roughly 26 000 orthogroups. A random split would put
near-identical orthologues on both sides. So we group first, and split whole
groups.

| split | orthogroups | what it is for |
|---|---|---|
| train | 23 622 | fits the weights |
| val | 1 056 | chooses an epoch *within* one run |
| spy | 327 | chooses *between* runs; opened once per comparison |
| test | 1 322 | opened once, at the end |

`val` and `spy` are disjoint and share no homologues. That matters more than
it sounds. The standard deviation of the evaluation statistic alone is 0.0064.
Selecting a checkpoint under a metric that noisy once picked a result 1.62
standard deviations above the truth. Choosing an epoch and choosing a model on
the same set inflates the number twice.

Both reference promoters sit entirely in `val`: the weak target `pks1` and the
strong housekeeping `tef1`. They are never trained on. They are visible to
epoch selection. They are absent from `spy` and from `test`.

**Choosing the epoch** follows a written rule, applied in a fixed order. We
decided that rule before we saw any number.

Two vetoes come first. The model must place `tef1` above `pks1`, or the scale
runs backwards at any correlation. And single-base substitutions must move the
prediction past the measured noise floor in more than a tenth of cases. A
model blind to single bases cannot serve a design task with an eight-base
budget.

Among the survivors we keep the epochs that `spy` cannot tell apart, within
one bootstrap half-width of the best. Then we take the highest `val` score
among those. The band is the whole point. The maximum of a noisy statistic
over thirty epochs is biased upward even when every epoch is identical.

Calibration is affine, and it is fitted on `spy`, never on `val`. It does not
change any rank correlation. It changes the level, which is what a user of the
predictions needs.

**Test was read once.** A script does it and refuses to run a second time. It
takes the epoch and the calibration from the selection artefact instead of
deciding them again, and it writes the checkpoint's SHA-256 next to the
number. That number is a test Spearman correlation of
**0.68051 ± 0.01749**.

---

## 7. What each input is worth

A channel being connected is not the same as a channel being used, so we
tested each one by taking it away from the trained model. We did that in two
different ways, and it is the pair that tells you something rather than either
number on its own:

* **Zeroing** replaces the channel with the model's learned "I don't know"
  token, and asks what it loses without the information.
* **Shuffling** hands the model another row's value instead, and asks whether
  it was ever reading the channel row by row in the first place.

If a channel were being used as a constant, zeroing it would hurt and
shuffling it would not.

The numbers below come from the checkpoint we actually ship, epoch 8, md5
`fd161127f0b11a0a7c33bf55e4bb2cfc`, scored on the validation split, where it
starts at **0.69481**.

There are two columns because a single one hides the answer. **Pooled** is the
accuracy over every measured cell, and it is dominated by the easiest
question: which gene is this. **Within-row** removes that. It subtracts each
row's own average across the conditions it was measured in, from both the
prediction and the truth, and correlates what is left, so it sees only the
part that depends on the growth condition. That part is 8.2 % of the variance,
across 5 003 rows.

| removed | pooled | Δ pooled | within-row | Δ within-row | gap |
|---|---|---|---|---|---|
| nothing | 0.69481 | — | 0.18680 | — | +0.1298 |
| experiment text | 0.69321 | −0.00160 | 0.18840 | **+0.00160** | +0.1290 |
| genomic flanks | 0.68721 | −0.00760 | 0.18444 | −0.00236 | +0.1342 |
| gene features | 0.66845 | **−0.02636** | 0.17835 | −0.00845 | +0.1517 |
| gene **and** flanks | 0.65295 | **−0.04186** | 0.17321 | −0.01359 | +0.1600 |
| strain, *shuffled* | 0.68866 | −0.00615 | 0.16782 | **−0.01898** | +0.1272 |
| flanks, *shuffled* | 0.67904 | −0.01578 | 0.18356 | −0.00324 | +0.1371 |
| gene, *shuffled* | 0.66037 | **−0.03444** | 0.18516 | −0.00164 | +0.1469 |

We checked that the within-row column measures what we think it does. If you
replace the prediction by its own row average, making it flat across the
conditions, which is exactly what a model ignoring the condition would be — the
score has to come out at zero. It comes out at **1.2 × 10⁻¹⁶**. Had a flat
prediction scored above zero, nothing in that column would mean anything.

<figure>
<svg viewBox="0 0 960 290" role="img" aria-label="A schematic of the conditioning path with the measured cost of each channel. Four channel boxes on the left - strain, experiment description, genomic flanks and gene features - each carry two numbers, the loss in the pooled score and the loss in the per-category score. The strain and the description are enclosed in one dashed box, because the description is keyed by genome and category and so is partly the strain; together they cost 0.012 pooled and 0.025 per-category. All four channels pass through per-channel encoders into a bottleneck of sixteen numbers, which drives a FiLM modulation of the pooled latent, which is the only thing the four prediction heads read." style="max-width:100%;height:auto;color:#1b1b1b;font-family:system-ui,-apple-system,Segoe UI,sans-serif">
  <defs>
    <marker id="cf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#d66a1a"/>
    </marker>
    <marker id="ck" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>

  <text x="112" y="24" font-size="13" font-weight="600">Channel</text>
  <text x="272" y="24" text-anchor="end" font-size="12.5" fill="#555">Δ pooled</text>
  <text x="342" y="24" text-anchor="end" font-size="12.5" fill="#555">Δ per-cat.</text>

  <!-- strain + description are ONE channel: the text is keyed by the cell -->
  <rect x="92" y="34" width="266" height="104" rx="6" fill="#fceee0" stroke="#8c3e0a" stroke-width="1.6" stroke-dasharray="5 4"/>

  <g fill="none" stroke="currentColor" stroke-width="1.6">
    <rect x="100" y="42" width="250" height="40" rx="5"/>
    <rect x="100" y="90" width="250" height="40" rx="5"/>
    <rect x="100" y="150" width="250" height="40" rx="5"/>
    <rect x="100" y="200" width="250" height="40" rx="5"/>
  </g>
  <g font-size="14">
    <text x="112" y="67">Strain</text>
    <text x="112" y="115">Description</text>
    <text x="112" y="175">Flanks</text>
    <text x="112" y="225">Gene</text>
  </g>
  <g font-size="13" text-anchor="end" font-variant-numeric="tabular-nums">
    <text x="272" y="67">−0.006</text>
    <text x="342" y="67" font-weight="700" fill="#8c3e0a">−0.022</text>
    <text x="272" y="115">−0.002</text>
    <text x="342" y="115" fill="#777">+0.002</text>
    <text x="272" y="175">−0.008</text>
    <text x="342" y="175" fill="#777">−0.002</text>
    <text x="272" y="225" font-weight="700" fill="#8c3e0a">−0.026</text>
    <text x="342" y="225" fill="#777">−0.008</text>
  </g>

  <text x="78" y="86" font-size="12.5" fill="#8c3e0a" transform="rotate(-90 78 86)" text-anchor="middle">One channel</text>

  <!-- everything converges on the bottleneck -->
  <g stroke="#d66a1a" stroke-width="1.8" fill="none" marker-end="url(#cf)">
    <path d="M358 62 C 392 62, 398 112, 424 120"/>
    <path d="M358 110 C 392 110, 400 118, 424 128"/>
    <path d="M358 170 C 392 170, 400 152, 424 144"/>
    <path d="M358 220 C 392 220, 398 164, 424 152"/>
  </g>

  <rect x="432" y="98" width="30" height="78" rx="5" fill="#fceee0" stroke="#d66a1a" stroke-width="2"/>
  <text x="447" y="142" text-anchor="middle" font-size="15" font-weight="700" fill="#d66a1a">16</text>
  <text x="447" y="194" text-anchor="middle" font-size="13" fill="#555">Bottleneck</text>

  <line x1="462" y1="137" x2="502" y2="137" stroke="#d66a1a" stroke-width="1.8" marker-end="url(#cf)"/>

  <rect x="508" y="106" width="172" height="62" rx="6" fill="none" stroke="#d66a1a" stroke-width="2"/>
  <text x="594" y="132" text-anchor="middle" font-size="15" font-weight="600" fill="#d66a1a">FiLM</text>
  <text x="594" y="154" text-anchor="middle" font-size="13" fill="#555">γ·latent + β</text>

  <!-- the sequence arrives from below: the condition modulates it, never replaces it -->
  <line x1="594" y1="238" x2="594" y2="174" stroke="currentColor" stroke-width="1.6" marker-end="url(#ck)"/>
  <text x="594" y="258" text-anchor="middle" font-size="13.5">Pooled Latent</text>

  <line x1="680" y1="137" x2="722" y2="137" stroke="currentColor" stroke-width="1.8" marker-end="url(#ck)"/>
  <rect x="728" y="106" width="192" height="62" rx="6" fill="none" stroke="currentColor" stroke-width="1.6"/>
  <text x="824" y="137" text-anchor="middle" font-size="16" font-weight="600">Heads</text>

</svg>
<figcaption><b>Figure 6.</b> What each conditioning channel is worth, measured
by removing it from the delivered model. <b>Δ pooled</b> is how much accuracy
is lost overall; <b>Δ per-cat.</b> is how much is lost on the part that depends
on the growth condition, which is only 8.2 % of the variance. The two columns
disagree about which channel matters, which is why we report both: the gene
features carry the overall score, the strain carries the condition-specific
part. The dashed box marks a catch. Descriptions are stored per
genome-and-category, so the description is partly the strain already, remove
one and the other still supplies most of the same information. Taken out
together they cost 0.012 overall and 0.025 on the condition-specific part,
which is more than the two add up to separately.</figcaption>
</figure>

Five things follow, and the fourth retracts a claim this document used to
make.

**Shuffling costs more than zeroing, on every channel.** Gene −0.034 against
−0.026, flanks −0.016 against −0.008. That is the signature of a channel read
*per row*: a wrong value is worse than an honest "I don't know", which is what
the learned unknown token exists to say. Had the two been equal, the channel
would have been acting as a constant and the model would not have been reading
it at all.

**The ordering is gene, then flanks, then strain, then text.** Removing gene
and flanks together costs **0.042**, more than removing each alone and adding
the two up (0.034), so the two channels reinforce rather than substitute for
each other.

**Removing conditioning makes the model overfit *more*, not less.** The
train − val gap grows from 0.130 to 0.160 as channels are taken away, and the
val column falls about four times faster than the train column. These channels
are doing work on the validation side that the trunk otherwise has to invent
from sequence alone.

**A missing description is free; a wrong one is not.** Replacing the text with
its unknown token moves the within-row score by **+0.0016**: nothing. Handing
the row *another row's* description costs **−0.0075**. The channel is being
read: what the first number measures is that the unknown token is a harmless
stand-in, not that the description is idle. Any claim about this channel built
on the zeroing row alone is built on the wrong probe, and an earlier draft of
this document made exactly that mistake in both directions.

**The per-category component is dominated by the strain, and only the
within-row column shows that.** Shuffling the strain costs **−0.019 to
−0.022** within-row across two runs against −0.006 pooled: it is the *largest*
per-category effect of any channel while being nearly the smallest overall
one. The two orderings genuinely disagree —

| ranked by | first | second | third | fourth |
|---|---|---|---|---|
| pooled | gene | flanks | strain | text |
| within-row | **strain** | gene | flanks | text *(negative)* |

, which is a (strain × category) effect: in this strain, carbon-source
experiments run higher than controls. It is the same structure the ridge
baseline picks up when the strain one-hot takes its within-row score from
0.105 to 0.394, and reporting only the pooled table hid it completely.

**Two notes on reading the table above.** The neighbouring epochs 7 and 9 give
the same ordering with the same signs: the deltas move, the conclusions do not
, so none of this is one epoch's noise. And the strain is shuffled rather than
zeroed on purpose: `species_id = 0` is a *real* strain, the first in the
registry, so zeroing it would have measured "what if every row were strain B10"
instead of "what does knowing the strain contribute".

**The read-out's columns are not ablatable, and that bounds the column.** The
cell read-out generates its seven columns by projecting the *category
centroids* through the shared `text_proj`, so the column directions are a
fixed part of the head; no ablation of a row's input can remove them, and a
channel reaches this component only through the pooled feature `h`. The within
-row column therefore measures how well a channel *steers* `h` along those
fixed directions, not whether it could invent them.

### The description and the strain are one channel, not two

Ablating them separately splits something that does not split, and the reason
is in the artefact: the experiment-text cache is keyed `<genome>|<category>`,
so the description **is** partly the strain. Measured on the cache the
delivered run actually loaded, over its 85 keys and 79 distinct vectors:

| factor the text vector is regressed on | η² |
|---|---|
| **(genome, category)** | **1.0000**, with 0 ambiguous cells |
| genome alone | **0.4134** |
| *control: genome labels shuffled* | *0.2177 ± 0.0158* |
| category alone | 0.2159 |

The first row is the structural fact: the vector is a deterministic function
of the cell, so the channel carries nothing a perfect (strain, category) pair
would not. The second says the strain alone accounts for **0.41** of its
variance, **+0.196** over the shuffle control. A single-channel ablation of
either one is therefore scored while its partner is still supplying much of
the same information.

So we removed them together, and read the interaction as
(joint) − (sum of singles):

| removed | Δ pooled | Δ within-row |
|---|---|---|
| description only, *zeroed* | −0.00160 | +0.00160 |
| description only, *shuffled* | −0.00224 | −0.00754 |
| strain only, *shuffled* | −0.00661 | −0.02230 |
| **description + strain** *(zero + shuffle)* | **−0.01160** | **−0.02482** |
| **description + strain** *(both shuffled)* | **−0.01011** | **−0.02810** |

**The strain was hiding what the description is worth.** With the strain
intact, zeroing the description costs 0.0016. With the strain already
shuffled, the same zeroing costs **0.0050**: **3.1× more**, and the pooled
interaction is −0.0034, i.e. super-additive. Shuffling instead of zeroing
gives the same direction, 1.6×. Read as one channel, the condition is worth
about **0.010 to 0.012 pooled and 0.025 to 0.028 within-row**, and those are
the figures to quote for "what the growth condition contributes", not the
single-channel rows.

**One place the story does not generalise, stated rather than smoothed.** The
masking result is clean in the pooled column for both probes. In the
within-row column it holds for zeroing but *reverses* for shuffling: a
shuffled description costs 0.0075 alone and 0.0058 once the strain is already
gone, which is slightly sub-additive. We do not have an account of that sign
change, and with the precision below it is close to the noise of the probe.

**Precision of the shuffle rows.** The permutation is drawn from one stream,
so a shuffle row depends on how many modes ran before it. Measured across two
runs of the same checkpoint, `strain, shuffled` came out −0.00615 and −0.00661
pooled, and 0.16782 and 0.16450 within-row: about **±0.0005 pooled and ±0.003
within-row**. Differences smaller than that between two shuffle rows are not
results. The zeroing rows are deterministic and carry no such spread.

### Where this component goes against us

A strain-aware ridge captures **more** of it than we do, and we would rather
put that here than have it found. Scored on identical rows, with the model's
better read-out:

| within-row score | |
|---|---|
| model, sequence and strain only *(4 583 rows)* | 0.178 |
| model, every channel on *(5 003 rows)* | 0.187 |
| `kmer4+species_ridge`, sequence and strain only *(4 583 rows)* | **0.394** |

The middle row is ours at its best; the first is the like-for-like one, since
the ridge also sees nothing but sequence and strain.

**2.2×, and it is not an artefact.** We checked the two obvious escapes and
neither holds: the row set contributes nothing (the component is 8.10% of cell
variance on these rows against 8.19% on the loop's, and the target deviation
agrees to four decimals), and turning every conditioning channel on buys only
about +0.009 of it. What separates the two models is their shape. The ridge has
**seven free per-category coefficient vectors**; this model has none and
generates its seven columns by projecting the category text through one shared
projection. That is the trade the architecture makes deliberately: free
columns cannot describe a condition they never saw, a text projection can —
and this number is its price. Section 9 says what the whole of section 7 does
and does not support.

### What the architecture earns over something simpler

The ablation says which inputs matter. It says nothing about whether the
machinery around them was worth building, so we trained baselines on the same
corpus, the same orthogroup-grouped split, the same seven-column cell target
and the same metric, selecting every hyperparameter on `val` exactly as the
model's own epoch is selected. The baselines see **sequence and strain and
nothing else**: no experiment text, no DNA-LM flanks, no orthology features,
no motif bank. Full corpus, 15 609 valid cells:

| model | parameters | pcc | scc |
|---|---|---|---|
| **HYPPE, full architecture** | 4 506 780 | **0.69369** | **0.67616** |
| CNN on one-hot + strain | 114 503 | 0.63730 | 0.61599 |
| 5-mer counts + strain, ridge | 7 308 | 0.55409 | 0.55280 |
| CNN on one-hot | 105 991 | 0.54783 | 0.52158 |
| 4-mer counts + strain, ridge | 1 932 | 0.54192 | 0.54040 |
| 3-mer counts + strain, ridge | 588 | 0.51528 | 0.51430 |
| 5-mer counts, ridge | 7 175 | 0.48090 | 0.47443 |
| 4-mer counts, ridge | 1 799 | 0.46420 | 0.45896 |
| 3-mer counts, ridge | 455 | 0.43212 | 0.42562 |

k-mers are **not** collapsed onto their reverse complement, because the model
does not canonicalise either and a baseline that did would be answering a
different question.

**The strain is the single largest factor, and it is not close.** Adding a
19-way one-hot moves the ridge by +0.078 at $k = 4$ and the CNN by +0.089.
Nineteen genomes differ enough in overall expression that knowing which one
you are in explains more than any extra sequence modelling these baselines can
do. Any future result on this corpus has to report whether it beat the
strain-only information, because a model can look good by learning the genome.

**Convolution is worth something real, but only at full data, and a subset
would have told us the opposite.** On the full corpus the CNN beats the best
bag-of-k-mers by +0.083. Repeated on 4 000 orthogroups the two **tie**
(0.51415 against 0.51511), with the CNN spending 59× the parameters. Anyone
reading an ablation in this project should check what fraction of the corpus
it ran on before believing its ordering.

**Having an output per category is not the same as knowing what the category
is.** Both CNN baselines score a *negative* $R^2$ on the condition main effect
(−0.363 and −0.126): with seven free output columns and no condition input
they do worse than predicting the per-category mean. The ridges, whose
per-category coefficients are explicit and linear, get the main effect right
but reach only 0.251 on that axis. HYPPE reaches 0.274, with the condition
arriving as a sentence embedding of the real experiment description rather
than as a column index.

**And the margin is the least informative thing in the table.** Split onto the
three axes the task actually has, one number becomes three movements in two
directions: **+0.056** on which gene, **+0.099** on which condition, and
**−0.081** on telling the nineteen strains apart. The architecture wins where
the task is sequence-to-expression and where the growth condition has to be
used, and it loses on strain discrimination: the loss that section 3 explains
and keeps.

**Two numbers for one model, and they are not interchangeable.** The 0.69369
above is the delivered checkpoint scored through the *baselines'* loader, so
that it meets them on identical rows. The 0.69481 the ablation deltas are
measured against is the same checkpoint through the *training loop's* own
evaluation pass, which also draws synthetic negatives. The gap is 0.0011 and
it is a loader difference, not a result: deltas belong to the second number
and cross-model comparisons to the first, and mixing them is the mistake this
project has made often enough to name.

The held-out-strain transfer experiment quoted in section 3, the
within-category contrast and the carbon cost of training are in
[ablations.md](ablations.md).

---

## 8. Choosing the ten promoters for the wet lab

The model is a predictor. A panel is a decision. This section is how we got
from one to the other.

<figure>
<svg viewBox="0 0 880 340" role="img" aria-label="Five search strategies each receive an identical budget of thirty thousand model evaluations and contribute a pool of candidates; the ranking head filters the pool against a two-sided acceptance window, which is drawn here in anchor units but is set on the calibrated scale, and ten designs are delivered." style="max-width:100%;height:auto;color:#1b1b1b">
  <defs>
    <marker id="a3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g font-family="system-ui,sans-serif" font-size="14">
    <rect x="12" y="20" width="176" height="32" rx="5" fill="none" stroke="currentColor"/><text x="100" y="41" text-anchor="middle">Random Substitution</text>
    <rect x="12" y="62" width="176" height="32" rx="5" fill="none" stroke="currentColor"/><text x="100" y="83" text-anchor="middle">Gradient + Beam</text>
    <rect x="12" y="104" width="176" height="32" rx="5" fill="none" stroke="currentColor"/><text x="100" y="125" text-anchor="middle">Genetic Algorithm</text>
    <rect x="12" y="146" width="176" height="32" rx="5" fill="none" stroke="currentColor"/><text x="100" y="167" text-anchor="middle">Masked Decoding</text>
    <rect x="12" y="188" width="176" height="32" rx="5" fill="none" stroke="currentColor"/><text x="100" y="209" text-anchor="middle">Discrete Diffusion</text>
  </g>

  <path d="M188 36 L226 114 M188 78 L226 114 M188 120 L226 118 M188 162 L226 122 M188 204 L226 126" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <line x1="226" y1="120" x2="258" y2="120" stroke="currentColor" stroke-width="1.5" marker-end="url(#a3)"/>

  <rect x="262" y="100" width="150" height="40" rx="6" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="337" y="125" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15">Candidate Pool</text>

  <line x1="412" y1="120" x2="444" y2="120" stroke="currentColor" stroke-width="1.5" marker-end="url(#a3)"/>
  <rect x="448" y="100" width="130" height="40" rx="6" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="513" y="125" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15">Hard Gates</text>

  <line x1="578" y1="120" x2="610" y2="120" stroke="currentColor" stroke-width="1.5" marker-end="url(#a3)"/>
  <rect x="614" y="100" width="118" height="40" rx="6" fill="none" stroke="#d66a1a" stroke-width="2"/>
  <text x="673" y="125" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15" fill="#d66a1a">Judge</text>

  <line x1="732" y1="120" x2="764" y2="120" stroke="currentColor" stroke-width="1.5" marker-end="url(#a3)"/>
  <rect x="768" y="100" width="104" height="40" rx="6" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="820" y="125" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15">10 Designs</text>

  <line x1="250" y1="290" x2="760" y2="290" stroke="currentColor" stroke-width="1.5"/>
  <line x1="330" y1="283" x2="330" y2="297" stroke="currentColor" stroke-width="2"/>
  <line x1="640" y1="283" x2="640" y2="297" stroke="currentColor" stroke-width="2"/>
  <text x="330" y="314" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13">pks1 = 0</text>
  <text x="640" y="314" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13">tef1 = 1</text>
  <rect x="330" y="276" width="310" height="28" fill="#d66a1a" opacity="0.10"/>
  <text x="485" y="271" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" fill="#d66a1a">MID must land here (≤ 16 edits)</text>
  <text x="712" y="271" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13">MAX above here (≤ 64)</text>
  <text x="250" y="243" font-family="system-ui,sans-serif" font-size="15" font-weight="600">The acceptance window</text>
</svg>
<figcaption><b>Figure 7.</b> Five strategies compete on an identical budget of
30 000 model evaluations from one fixed seed. Each hands the Judge the whole
pool of candidates it collected along its path, not just its own best one. The
Hard Gates are absolute: no ATG in the last 10 bp, GC and longest-homopolymer
inside the band measured on real promoters of the same genome, and a floor on
the realness score. The Judge is a <i>different</i> model from the one that
proposed, fine-tuned on anchor pairs, and acceptance is two-sided, so a MID
design can fail for being too strong. The window is drawn in anchor units,
with pks1 at 0 and tef1 at 1, but it is set on the calibrated scale. The ten
designs ship with both native anchors beside them.</figcaption>
</figure>

**Two anchors set the scale.** `pks1` is the promoter we want to replace.
`tef1` is a strong constitutive promoter of the same strain. We *report*
every prediction in *anchor units*, where 0 is `pks1` and 1 is `tef1`,
because an absolute predicted value has no biological unit attached to it
and anchor units need no trust in our scale.

Reporting and deciding are not the same step, and here they use different
scales. Anchor units are built from the model's own predicted gap between
the two anchors, and that gap is wrong: the model spreads the anchors much
further apart than the measurement does, so a promoter can look comfortably
"between the anchors" in anchor units while sitting nowhere near the middle
in reality. The *acceptance window* is therefore set on the calibrated
scale, not in anchor units. Section 10.2 gives the measurement that forced
that split.

**Two design briefs.** *MID* asks for a promoter between the two anchors,
changing at most 16 of the 800 bases. *MAX* asks for one above `tef1`, with at
most 64. The budgets are small on purpose. A surgical edit keeps everything
else about the promoter intact. That makes the result interpretable, and the
construct easier to trust.

**Five strategies, one budget.** Random substitution is the control a guided
method has to beat. The other four are gradient-guided editing with beam
search, a genetic algorithm, masked iterative decoding, and discrete
diffusion. Each one received exactly 30 000 model evaluations, from one fixed
seed. An earlier comparison had given them between 1 920 and 30 000, so it was
measuring compute rather than method.

The genetic algorithm deserves a paragraph, because its representation is what
makes the edit budget trustworthy. An individual is not a sequence. It is a
**set of (position, base) substitutions relative to `pks1`**. The budget is
the length of the individual, so it cannot be exceeded by construction rather
than by a penalty term. Crossover is uniform over the two parents' sets, with
the shared core preserved. Mutation adds, removes or moves one substitution.
The method took two of the ten delivered slots.

**Scoring happens on decoded DNA, never on codes.** This is a measured
constraint, not a preference. Substituting the whole coarse level moves the
code-space prediction a great deal and changes *not one base*. Substituting
the finest level changes hundreds of bases and barely moves it. A search
guided by the code-space score would optimise a number the acceptance gate
never sees.

**The proposer and the judge are different models.** The regression head
proposes. A ranking head, fine-tuned on anchor pairs, decides membership.

On the raw model the "is this above `pks1`" decision was near chance, at
balanced accuracy 0.5669 on the 299 validation promoters of P1. The model had
been trained to order promoters *within* families and *across* the corpus. It
was never trained on the anchor comparison the panel is accepted on.
Fine-tuning the ranking head alone fixed it, and the numbers come out of the
shipped judge itself, which stores them (`results/judge_boundary.json`):

| on validation, P1 | before | after |
|---|---|---|
| balanced accuracy, "above `pks1`" | 0.5669 | **0.8138** |
| Spearman on the held-out `pks1` orthologues | −0.4857 | **+0.8857** |
| Spearman on the corpus as a whole | 0.5688 | 0.5135 |
| fraction of validation below `tef1` | 0.9900 | 0.9967 |

The boundary this panel is accepted on went from a coin flip to 0.81, and the
family ranking went from *backwards* to +0.89: the batch keeps a quarter of
its pairs on family comparisons precisely so that gain is not lost. What was
paid for it is the last row but one: general corpus ranking slipped by 0.055.
That is the trade, and it is why the proposer and the judge are two models
rather than one. Their agreement at the chosen point
is then a result rather than an assumption. Across the ten designs their rank
correlation was 1.00, with a single adjacent swap.

**Hard gates every candidate must pass.** No ATG in the last 10 bases: that
region is the 5′ UTR immediately before the start codon, and an ATG there
creates an upstream open reading frame. GC content and longest homopolymer
must sit inside the band measured on real promoters of the same strain. A
realness floor comes from the discriminator. The upstream-ATG count further
from the start is *reported, not enforced*. Calling an ATG a uORF requires
knowing where transcription starts, and that is not resolved for this locus.

**What shipped, including the part that did not go to plan.** Ten designs:
five methods × two briefs. Both native anchors travel in the panel, so the
wet lab can place everything on the scale the anchors define.

Six of the ten met the edit budget. Three MID designs changed 5, 16 and 16
bases, and three MAX designs changed 59, 63 and 63. They came from
gradient-guided editing, the genetic algorithm and masked decoding.

**The other four did not, and the panel labels them so.** Random substitution
and discrete diffusion never reached either brief inside the budget, so they
are delivered in a second, unconstrained regime, marked `MID_free` and
`MAX_free` in `data/final_promoters.tsv`. Those four changed 612, 580, 578
and 577 bases and retain 23 to 28 per cent identity to `pks1`. At that
distance they are not edits of `pks1` in any useful sense; they are new
sequences that happen to start from it, about as far from the target as
`tef1` is, which differs at 616 positions.

We ship them because a panel of five methods that quietly drops the two that
failed is not a comparison of five methods. One of the two is the random
control, and a control that cannot meet the brief is the control working. The
other is discrete diffusion, which generates a sequence rather than editing
one, so a budget of sixteen bases is not a natural constraint for it. We
record that as the observation it is, not as a diagnosis: we did not run the
experiment that would separate "the method cannot do this" from "our
implementation of it could not".

### The panel as delivered

Strength is given in anchor units, where `pks1` is 0 and `tef1` is 1. The
last column is the score of the ranking head, the judge, which is a separate
model from the one that proposed the sequence.

| design | method | brief | edits | identity to `pks1` | strength | judge |
|---|---|---|---|---|---|---|
| `pks1` | native anchor | n/a | 0 | 1.000 | 0.000 | 0.31 |
| `tef1` | native anchor | n/a | 616 | 0.230 | 1.000 | 10.75 |
| MID, gradient | gradient + beam | MID | 16 | 0.980 | 0.694 | 5.52 |
| MID, masked | masked decoding | MID | 16 | 0.980 | 0.492 | 0.83 |
| MID, genetic | genetic algorithm | MID | 5 | 0.994 | 0.011 | 0.63 |
| MAX, gradient | gradient + beam | MAX | 59 | 0.926 | 1.121 | 12.10 |
| MAX, masked | masked decoding | MAX | 63 | 0.921 | 1.106 | 11.82 |
| MAX, genetic | genetic algorithm | MAX | 63 | 0.921 | 1.101 | 11.72 |
| MID, random | random substitution | MID, free | 612 | 0.235 | 0.584 | 2.84 |
| MID, diffusion | discrete diffusion | MID, free | 580 | 0.275 | 1.060 | 10.68 |
| MAX, random | random substitution | MAX, free | 578 | 0.278 | 0.636 | 4.39 |
| MAX, diffusion | discrete diffusion | MAX, free | 577 | 0.279 | 1.055 | 11.46 |

Three things in that table are worth reading rather than skipping.

**Two designs miss their own brief, and both are in the free regime.** MID
asks for a promoter between the anchors and MAX for one above `tef1`. The
diffusion MID lands at 1.060, above `tef1`; the random MAX lands at 0.636,
below it. Every design that kept to the edit budget met its brief.

**The genetic MID is technically inside the window and practically at the
floor.** It changed five bases and moved the prediction to 0.011 anchor
units, which is `pks1` with a rounding error. It is in the panel as the
smallest edit we could make that the judge still accepted, not as a design we
expect to measure apart from the native promoter.

**The proposer and the judge agree, and that is a result rather than an
assumption.** Ranking the ten designs by the proposer and by the judge gives
the same order except for one adjacent swap, between the two diffusion
designs. They are separate models, so the agreement is evidence that
acceptance is not an artefact of whichever head happened to score.

**Each design is scored twice, once per construct.** Our wet lab builds the
panel in two plasmid contexts, `pAN7-1` and `pHYPPE`, and
`data/final_promoters.tsv` carries a separate mean and standard deviation for
each. The table above shows the `pAN7-1` column.

The two do not differ by a constant. `pHYPPE` sits lower for the nine
stronger rows and slightly higher for the two weakest, so the second context
compresses the panel toward its middle rather than shifting it. We report
that as an observation about the model, not about the plasmids: both
contexts reach the model through the same learned "unknown" flank token, so
a difference between them is the model's own uncertainty about an
out-of-distribution neighbourhood, not knowledge of either vector. The
constructs, and the measurement that will test these predictions, belong to
the wet lab pages rather than to this one.

---

## 9. What these numbers claim, and what they do not

We would rather state these ourselves than have them found.

**The predictions are not fluorescence.** Three domain shifts sit between our
training label and a wet read-out: mRNA to protein, native locus to plasmid,
and rank within a library to one absolute value. The model was trained on none
of them.

**The absolute scale is wrong where we could check it.** Measured, `pks1` sits
at the 94th percentile of the promoters of its strain. Our model places it
near the 13th. "Stronger than `pks1`" therefore means stronger than *where the
model puts* `pks1`. That is exactly why a judge fine-tuned on that boundary
decides membership, and why both anchors travel with the panel.

**MAX is extrapolation.** Asking for a promoter above the strongest anchor
asks for a prediction outside the range the judge was calibrated on. The panel
table labels those rows as such.

**The levels are not equal partners.** The coarse level dominates the
prediction and never reaches the decoder. The fine levels build the sequence
and barely move the prediction. Equalising that leverage is the main open
problem we carry forward.

**The latent describes rather than abstracts.** At 1.99× the sequence's own
entropy, the hierarchy carries the promoter almost literally. That is good for
decoding. It is bad for any claim that the model has found regulatory grammar.

**Design time is zero-shot.** No measurement from the target strain entered
the pipeline. The strain enters by its learned position on the phylogeny. The
condition enters by its written description. The plasmid context enters by a
learned "unknown" token, because a plasmid neighbourhood sits 5.1 standard
deviations outside the distribution of genomic flanks the model saw in
training.

---

## 10. Engineering cycles: what we built wrong, and how we found out

Everything below is something we shipped, or nearly shipped, followed by the
measurement that caught it. We report them because the measurement is the
transferable part. In almost every case the broken version **ran, logged
sensible numbers, and looked finished**. So the fix was never "notice the
bug". It was "build the observable that would have made it visible".

### 10.1 Architecture

**The latent could not carry the sequence it was supposed to rebuild.**
Our first design was a three-level code of about 1 500 bits. Its
reconstruction sat at 0.8888, which is 89 of 800 positions wrong. We read that
as undertraining for longer than we should have. The arithmetic says
otherwise. An 800 bp sequence carries roughly 1 600 bits, so a 1 500-bit
channel runs at **0.94×** of what it must describe, and 0.8888 is a *ceiling*
rather than a training failure. The experiment meant to settle it could not
have: both of its arms, 1 300 and 1 500 bits, sat below 1 600.
*Changed:* five levels and 3 186 bits, **1.99×**. Reconstruction from codes
alone then became essentially exact.
*Learned:* for a model that generates by decoding, the capacity question comes
**before** the training question. We now quote the ratio next to the
reconstruction number every time, because the first explains the second.

**Thirteen of twenty strains sat at the same point.**
Strain conditioning is read off a phylogeny. The default tree had **10 leaves,
named by accession**, while the strain registry has **20**. Thirteen strains
therefore collapsed onto a single coordinate. Among them were six of the seven
strains that share the `pks1` promoter **byte for byte**, while their measured
expression spans **0.039 to 7.624**, a factor of 195. The channel was not
dead. It was structurally unable to express the one contrast we most needed,
and no loss curve would have said so.
*Changed:* a 20-leaf in-house NJ tree whose leaf names are exactly the
registry's. The falsifiers were checked **before** the swap, not after.
Rerooting leaves the patristic distances unchanged (0.000e+00). Agreement with
an independent reference tree is Pearson **+0.9953** over 21 shared pairs. The
thirteen formerly-collapsed strains span **98.7%** of the new tree's range.
Collapsed pairs fell from **78/190 to 3/190**. The three that remain are not
resolved by 439 protein markers, and they are separate learned rows anyway.

**The prior held hardest exactly the pairs that had to separate.**
A prior pulls the strain embeddings toward patristic distance. Written as a
relative error, its stiffness grows as $1/d^2$. Across 182 admissible pairs the
median stiffness is 6.04 and the maximum 5.244e+04. The seven
`pks1`-identical strains are the closest pairs on the tree, so **17 of their
21 pairs carry a median stiffness of 4.283e+04, which is 7 089× the median**,
while the expression they must separate differs 195-fold. The regulariser was
pulling hardest against the signal.
*Changed:* a floor on the **denominator**, `genome_prior_denom_floor: 0.005`,
and not on the numerator, so the target stays
$\text{scale}\cdot d_{\text{patr}}$. The value is not tuned. 0.005 is where
the marker matrix's own relative precision reaches 20% (SE 9e-4). Sixteen of
182 pairs softened, and the max-to-median stiffness ratio went
**8 679× → 1 375×**.
A separate hard floor is needed for the prior to run at all. The NJ tree
contains a pair at $d = 10^{-6}$, and without `PRIOR_MIN_DIST = 10^{-3}` the
loss reaches 180 and freezes the encoder.

**Two branches that were declared, configured, and unreachable.**
`GeneEncoder`, 389 lines, was **never instantiated anywhere** in the source.
Setting `conditioning.gene.enabled: true` changed exactly one quantity,
`context_dim()`, which the token path does not read. The call site passed its
features behind a guard that could never be true. Separately,
`heads.decomposition` built **228 483 parameters, 5.5% of the model**. It ran
them forward, applied weight decay to them, and wrote them into every
checkpoint, while the training loop never computed a loss for them.
*Changed:* the gene channel was wired as a fourth conditioning token, and
admitted **only on measurement**. That was a ridge on 44 phyletic columns,
10 168 training orthogroups against 1 155 held out by orthogroup: R²
**+0.3163** versus a shuffled control of **−0.0069 ± 0.0033**. The
decomposition heads were switched off until their data pipeline exists.
*Learned:* a configuration key that looks enabled is not a mechanism. Our
acceptance gates now assert on the **built model**. They check parameter
counts, which blocks carry the module, and whether the output actually differs
from the baseline. An exactly zero difference is the signature of a mechanism
that is present and inert.

**The head meant to teach the strain channel could not teach it.**
Pairs are drawn within one genome, so the strain is **constant inside a pair**
and the ranking head teaches that channel nothing. Its only teacher was the
absolute head, which is the one that overfits: `task_absolute` falls 1.041 →
0.367 while validation never beats epoch 2 for 23 epochs.
*Changed:* `ortholog_fraction: 0.30` reserves part of every batch for
orthogroup blocks, which hold one gene and one category across several
genomes. Supply was never the constraint: 9 098 cells, 1.2 M admissible pairs,
91.7% above the gap threshold. **Arrival** was the constraint: 0.60 such pairs
in a random batch, against 5.1 under the sampler.

### 10.2 Literature and assumptions

**"Biology is hierarchical, therefore hyperbolic."**
This is the usual motivation for hyperbolic embeddings, and we adopted it
before testing it. We measured Gromov's δ on a regulatory-sequence corpus and
on a motif bank, both in the sibling track of this codebase, not on the
*Trichoderma* corpus. Regulatory sequence is not meaningfully more tree-like
than random points. The motif space came out **less** tree-like than random
Euclidean points of the same dimension.
*Changed:* we kept the geometry and **discarded the argument**. What is a tree
here is the code hierarchy we impose, by construction. That is the only
justification we now claim, and it is the one that survives the measurement.

**A published method, applied as published, starved the task we cared about.**
Multi-task bargaining (Navon et al., ICML 2022) equalises utilities across
tasks. So it awards the **largest** weight to the player with the **weakest**
gradient, and a nearly solved task has the weakest gradient. We measured this
on a sibling model in this codebase. Reconstruction took **2.31 of a budget of
3.0**, while the two prediction heads were suppressed to **0.39 and 0.31**,
for a task already sitting at 0.9998.
*Changed:* explicit normalised priors (1.5 / 1.0 / 0.6), and reconstruction
demoted from a player to a **constraint** held at a target by a Lagrange
multiplier.
*Learned:* a method's guarantee is about the objective it states. Ours was not
that objective, and the solver was doing its job correctly.

**The default ranking loss stops learning while still wrong.**
A margin hinge has **exactly zero** gradient once a pair is correctly ordered.
So it is indifferent to *how* wrong the gap is. Measured on a single correctly
ordered pair, as the gap widens:

| gap | hinge, $m = 0.1$ | hinge, $m = 0$ | logistic |
|---|---|---|---|
| 0.05 | −1.000e+00 | **−0.000e+00** | −4.875e-01 |
| 0.50 | **−0.000e+00** | **−0.000e+00** | −3.775e-01 |
| 3.00 | **−0.000e+00** | **−0.000e+00** | −4.743e-02 |

The second column is the part that caught us. `margin: 0.0` is **not** a
marginless loss. It is the hinge that gives up earliest.
*Changed:* the logistic (Bradley–Terry) form, whose gradient stays
proportional to how wrong the gap is. Coupled to it, epoch selection moved to
PCC rather than Spearman, because a metric that reads gaps needs a loss that
reads gaps.

**We assumed the model had a start-codon prior. It has none, and the data
cannot teach it one.**
We expected the model to avoid introducing upstream ATG. It has no mechanism
to do so. No buffer, parameter or loss term in the model mentions ATG, uORF or
the start codon, and the only positional information is a single
parameter-free ramp. Measured, substituting a triplet to ATG moves the
prediction by **−0.058**, against **−0.053** for five composition-matched
controls. That is, by nothing.
*Changed:* the constraint moved to an **external gate** in the design loop.
The reason it does not belong in the loss is biological rather than technical.
Real promoters are full of ATG, and `pks1` itself carries 14, so there is
nothing in the data to learn from. The prohibition comes from our construct,
not from the organism.

**`pks1` is not a weak promoter, and finding that out moved the panel.**
We had treated it as the weak anchor. Measured, its control value is **7.355,
the 94th percentile** of 10 466 promoters in that strain, and 230 of 300
held-out promoters of that strain sit below it. The model compounds the error
in the other direction. It predicts an anchor gap of **8.441** where the
measurement says **4.845**. So in anchor units **87%** of promoters appear to
fall "between `pks1` and `tef1`", against **6%** in the data.
*Changed:* the MID acceptance window is derived on the **calibrated** scale
rather than in anchor units. Both anchors are carried in the panel itself, so
the wet lab can place every result without trusting our scale at all.

**One phrase, two different measurements.**
"The gene channel" named two things. One was a ridge on phyletic features
predicting the **label**, at +0.3163, which is why we built it. The other was
a ridge on codon-usage features predicting the **model's residual**, at
−0.1279, which says not to bother. Different features, different target,
opposite conclusions. For a while both numbers were in play under one name.
They are now always qualified by what they predict.

### 10.3 What the cycle taught us as a method

Three failure modes recur across every item above. All three fail **silently**
rather than loudly:

1. **A number measured on one object, quoted about another.** A threshold
   calibrated on random input and applied to real sequence. A figure from a
   sibling model quoted as ours. A training-mode statistic compared against an
   evaluation-mode one.
2. **A mechanism that is built, configured, paid for and never reached.** The
   gene encoder. The decomposition heads. A key whose first occurrence in a
   file is not the one the parser uses.
3. **A gate that cannot fail.** A threshold already satisfied at
   initialisation, or a check whose own bug makes it pass.

What changed in response is procedural, and it is the part we would carry to
any next project. Thresholds are written down **before** the run that tests
them. Every gate is given a positive control that must make it fail, and a
gate we cannot make fail is not evidence. And any measurement used to
calibrate a run is taken **in the regime the run will use**, because a network
in training and the same network in evaluation are not the same function.

---

## 11. Reproducing this

Everything above runs from one seed. The model, the training loop, the split
builder, the synthetic negatives and the diagnostics are all in the
repository. `bash examples/smoke/run.sh` trains a small model on a synthetic
corpus and runs the acceptance suite, with no data and no GPU required. The
suite's strongest check compares 174 evaluation metrics before and after a
checkpoint round trip, at tolerance *zero*.

### What ships with this repository

Enough to rebuild the delivered model and to check every claim on this page
without asking us for anything.

| | |
|---|---|
| trained weights | `weights/`, two checkpoints — see below |
| the motif bank | `data/motifs.meme`, all 252 frozen profiles |
| the split | `data/splits/`, every orthogroup assignment |
| the delivered panel | `data/final_promoters.fasta` and `.tsv` |
| conditioning artefacts | the 20-leaf tree, the condition taxonomy, the category text vectors |
| orthology tables | `data/final_datasets/orthology/` |

### Which checkpoint every number on this page came from

Two checkpoints ship, and they do different jobs, so it is worth saying plainly
which one each figure was measured on.

| checkpoint | md5 | what it is | what we report from it |
|---|---|---|---|
| `weights/hyppe_foundational.pt` | `fd161127…2cfc` | the foundation model, epoch 8 | everything except the panel: the bit budget and ladder geometry in section 4, the strain space in section 3, all of section 7, and the test score of 0.68051 |
| `weights/hyppe_judge.pt` | `906bd7a0…58b6` | the same model with its ranking head fine-tuned on anchor pairs | the acceptance decision for the ten designs in section 8, and the four boundary figures there |

**The judge really is the same trunk.** We did not take that on trust from the
training script — we compared the two files tensor by tensor. Of 471 tensors,
**465 are identical bit for bit**, six differ, and all six are inside
`pairwise.*`: the two trunk layers of the ranking head and its output layer.
Nothing outside the ranking head moved, which is what "the proposer and the
judge share an encoder" has to mean if the panel's acceptance is to say
anything about the model that proposed.

Two wrinkles to know if you inspect the files yourself. The judge still carries
the parent's `name` and `epoch` in its metadata, and its stored *training*
metrics are the foundation model's, flagged `metrics_stale`, so the two cannot
be told apart by their headers, only by md5 or by the six tensors above. Its
own fine-tuning metrics are there, but under a separate key, `syjam_kotwice`,
which also records that the weight change outside `pairwise.*` was exactly 0.0,
the seed (42) and the step it stopped at (225).

The table above is read out of the shipped file rather than quoted from any
script or draft. An earlier version of this page reported 0.54 → 0.87, which
matches no run at all: boundary figures copied from a training script describe
whichever run its author last looked at, not the checkpoint that ships.

What is not here is the training corpus itself, the promoter sequences paired
with expression. It is assembled from public archives, so none of it is ours
to withhold or to give: the RNA-seq runs come from ENA and NCBI under their
own accessions, and a companion pipeline turns them into the table we train
on. That pipeline and the one that builds the motif bank are released
separately by our team. Anyone can therefore rebuild the corpus from the
accessions rather than take our copy of it on trust, which is the stronger
form of reproducibility and the reason we did not simply dump a CSV.

For a reader who wants to run something now rather than rebuild a corpus
first, `bash examples/smoke/run.sh` trains a small model end to end on a
synthetic corpus and runs the full acceptance suite, with no data, no
downloads and no GPU.
