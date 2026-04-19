const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'Blockwise矩阵求逆算法原理';
pres.author = 'LU-inverse';

// Color palette - Midnight Executive theme
const NAVY = "1E2761";
const ICE_BLUE = "CADCFC";
const WHITE = "FFFFFF";
const DARK_TEXT = "1E293B";

// Slide 1: Blockwise & Schur Complement Algorithm
let slide1 = pres.addSlide();
slide1.background = { color: NAVY };

// Title
slide1.addText("Blockwise矩阵求逆与Schur补算法", {
  x: 0.5, y: 0.3, w: 9, h: 0.8,
  fontSize: 36, fontFace: "Arial", bold: true,
  color: WHITE, align: "center"
});

// Left column - Algorithm principle
slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.3, y: 1.2, w: 4.7, h: 4.0,
  fill: { color: WHITE, transparency: 5 },
  rectRadius: 0.15
});

slide1.addText([
  { text: "分块求逆原理", options: { bold: true, fontSize: 20, color: NAVY, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "将 n×n 矩阵分为 2×2 分块:", options: { fontSize: 14, color: DARK_TEXT, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "  [A₁₁  A₁₂]", options: { fontSize: 13, color: "4472C4", breakLine: true } },
  { text: "  [A₂₁  A₂₂]", options: { fontSize: 13, color: "4472C4", breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Schur补公式:", options: { fontSize: 14, color: DARK_TEXT, bold: true, breakLine: true } },
  { text: "  S = A₂₂ - A₂₁·A₁₁⁻¹·A₁₂", options: { fontSize: 13, color: "4472C4", breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "逆矩阵分块:", options: { fontSize: 14, color: DARK_TEXT, bold: true, breakLine: true } },
  { text: "  左上: A₁₁⁻¹ + A₁₁⁻¹·A₁₂·S⁻¹·A₂₁·A₁₁⁻¹", options: { fontSize: 12, color: "4472C4", breakLine: true } },
  { text: "  右上: -A₁₁⁻¹·A₁₂·S⁻¹", options: { fontSize: 12, color: "4472C4", breakLine: true } },
  { text: "  左下: -S⁻¹·A₂₁·A₁₁⁻¹", options: { fontSize: 12, color: "4472C4", breakLine: true } },
  { text: "  右下: S⁻¹", options: { fontSize: 12, color: "4472C4", breakLine: true } },
], { x: 0.5, y: 1.35, w: 4.3, h: 3.8 });

// Right column - Advantages
slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 5.0, y: 1.2, w: 4.7, h: 4.0,
  fill: { color: WHITE, transparency: 5 },
  rectRadius: 0.15
});

slide1.addText([
  { text: "算法优势", options: { bold: true, fontSize: 20, color: NAVY, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "分治策略", options: { bullet: true, fontSize: 14, color: DARK_TEXT, bold: true, breakLine: true } },
  { text: "  将大问题分解为小规模求逆", options: { fontSize: 12, color: "64748B", breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "缓存友好", options: { bullet: true, fontSize: 14, color: DARK_TEXT, bold: true, breakLine: true } },
  { text: "  子块尺寸匹配CPU缓存行", options: { fontSize: 12, color: "64748B", breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "可并行化", options: { bullet: true, fontSize: 14, color: DARK_TEXT, bold: true, breakLine: true } },
  { text: "  多个子块求逆可独立执行", options: { fontSize: 12, color: "64748B", breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "递归扩展", options: { bullet: true, fontSize: 14, color: DARK_TEXT, bold: true, breakLine: true } },
  { text: "  可递归应用于更大矩阵", options: { fontSize: 12, color: "64748B", breakLine: true } },
], { x: 5.2, y: 1.35, w: 4.3, h: 3.8 });

// Slide 2: Performance Comparison
let slide2 = pres.addSlide();
slide2.background = { color: "F5F7FA" };

// Title with accent bar
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.08,
  fill: { color: NAVY }
});

slide2.addText("性能对比: LAPACK vs Eigen vs Blockwise", {
  x: 0.5, y: 0.3, w: 9, h: 0.6,
  fontSize: 32, fontFace: "Arial", bold: true,
  color: NAVY, align: "center"
});

// Three method cards
const cardY = 1.1;
const cardH = 2.2;
const cardW = 2.9;

// LAPACK Card
slide2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 0.4, y: cardY, w: cardW, h: cardH,
  fill: { color: WHITE },
  rectRadius: 0.1,
  shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 }
});
slide2.addText([
  { text: "LAPACK", options: { bold: true, fontSize: 18, color: NAVY, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "方法: LU分解 + 求逆", options: { fontSize: 12, color: DARK_TEXT, breakLine: true } },
  { text: "接口: dgetrf + dgetri", options: { fontSize: 12, color: DARK_TEXT, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "40×40矩阵:", options: { fontSize: 12, color: "64748B", bold: true, breakLine: true } },
  { text: "  时间: 21μs", options: { fontSize: 13, color: "E11D48", breakLine: true } },
  { text: "  残差: ~1e-13", options: { fontSize: 12, color: DARK_TEXT, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "特点: 稳定可靠", options: { fontSize: 11, color: "64748B", italic: true, breakLine: true } },
], { x: 0.55, y: cardY + 0.15, w: cardW - 0.3, h: cardH - 0.3 });

// Eigen Card
slide2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 3.55, y: cardY, w: cardW, h: cardH,
  fill: { color: WHITE },
  rectRadius: 0.1,
  shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 }
});
slide2.addText([
  { text: "Eigen", options: { bold: true, fontSize: 18, color: NAVY, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "方法: C++模板库", options: { fontSize: 12, color: DARK_TEXT, breakLine: true } },
  { text: "接口: Matrix::inverse()", options: { fontSize: 12, color: DARK_TEXT, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "40×40矩阵:", options: { fontSize: 12, color: "64748B", bold: true, breakLine: true } },
  { text: "  时间: 16μs", options: { fontSize: 13, color: "0D9488", breakLine: true } },
  { text: "  残差: ~7e-14", options: { fontSize: 12, color: "0D9488", bold: true, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "特点: 精度最高", options: { fontSize: 11, color: "64748B", italic: true, breakLine: true } },
], { x: 3.7, y: cardY + 0.15, w: cardW - 0.3, h: cardH - 0.3 });

// Blockwise Card - highlighted
slide2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 6.7, y: cardY, w: cardW, h: cardH,
  fill: { color: "E0F2FE" },
  rectRadius: 0.1,
  shadow: { type: "outer", blur: 6, offset: 3, color: NAVY, opacity: 0.15 }
});
slide2.addText([
  { text: "Blockwise", options: { bold: true, fontSize: 18, color: NAVY, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "方法: 2×2分块 + Schur补", options: { fontSize: 12, color: DARK_TEXT, breakLine: true } },
  { text: "混合: Eigen + BLAS", options: { fontSize: 12, color: DARK_TEXT, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "40×40矩阵:", options: { fontSize: 12, color: "64748B", bold: true, breakLine: true } },
  { text: "  时间: 15μs ★", options: { fontSize: 13, color: "2563EB", bold: true, breakLine: true } },
  { text: "  残差: ~1e-13", options: { fontSize: 12, color: DARK_TEXT, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "特点: 速度最快", options: { fontSize: 11, color: NAVY, italic: true, breakLine: true } },
], { x: 6.85, y: cardY + 0.15, w: cardW - 0.3, h: cardH - 0.3 });

// Large scale comparison section
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0.4, y: 3.5, w: 9.2, h: 1.8,
  fill: { color: WHITE, transparency: 50 },
  shadow: { type: "outer", blur: 3, offset: 1, color: "000000", opacity: 0.08 }
});

slide2.addText([
  { text: "大规模矩阵性能对比", options: { bold: true, fontSize: 16, color: NAVY, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "n=200:  Blockwise 443μs  vs  Eigen 2111μs  →  加速 ", options: { fontSize: 13, color: DARK_TEXT, breakLine: true } },
  { text: "4.77x", options: { fontSize: 13, color: "2563EB", bold: true, breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "n=500:  Blockwise 4.4ms  vs  Eigen 19.5ms  →  加速 ", options: { fontSize: 13, color: DARK_TEXT, breakLine: true } },
  { text: "4.45x", options: { fontSize: 13, color: "2563EB", bold: true, breakLine: true } },
], { x: 0.6, y: 3.6, w: 8.8, h: 1.6 });

// Conclusion box
slide2.addText([
  { text: "结论: ", options: { bold: true, fontSize: 14, color: NAVY } },
  { text: "小规模(n≤50)优先Eigen(精度高); 大规模(n≥200)优先Blockwise(快4x+)", options: { fontSize: 13, color: DARK_TEXT } },
], { x: 0.5, y: 5.35, w: 9, h: 0.4, align: "center" });

pres.writeFile({ fileName: "/Users/yingwei/Documents/code/testcode/solver/LU-inverse/matrix_inverse_algorithm.pptx" });
console.log("PPTX created: matrix_inverse_algorithm.pptx");