// The yHatTrick mark: "ŷ" — a y wearing the regression "hat" (ŷ, the predicted value).
// Doubles as the favicon (see app/icon.svg) and the first glyph of the wordmark.
export default function BrandMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-hidden="true">
      <defs>
        <linearGradient id="bm-g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#4a90d9" />
          <stop offset="1" stopColor="#2f6cb0" />
        </linearGradient>
      </defs>
      <rect width="100" height="100" rx="22" fill="url(#bm-g)" />
      <polyline
        points="33,43 50,28 67,43"
        fill="none"
        stroke="#ffffff"
        strokeWidth="8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <text
        x="50"
        y="80"
        textAnchor="middle"
        fontFamily="Helvetica, Arial, sans-serif"
        fontSize="58"
        fontWeight="700"
        fill="#ffffff"
      >
        y
      </text>
    </svg>
  );
}
