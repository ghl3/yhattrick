// Team abbreviation -> city + nickname, plus the NHL logo CDN. Covers every abbrev that appears in
// our covered seasons (incl. ARI and UTA). Logos come from the NHL asset CDN by abbreviation.

export const TEAM_INFO: Record<string, { city: string; name: string }> = {
  ANA: { city: "Anaheim", name: "Ducks" },
  ARI: { city: "Arizona", name: "Coyotes" },
  BOS: { city: "Boston", name: "Bruins" },
  BUF: { city: "Buffalo", name: "Sabres" },
  CAR: { city: "Carolina", name: "Hurricanes" },
  CBJ: { city: "Columbus", name: "Blue Jackets" },
  CGY: { city: "Calgary", name: "Flames" },
  CHI: { city: "Chicago", name: "Blackhawks" },
  COL: { city: "Colorado", name: "Avalanche" },
  DAL: { city: "Dallas", name: "Stars" },
  DET: { city: "Detroit", name: "Red Wings" },
  EDM: { city: "Edmonton", name: "Oilers" },
  FLA: { city: "Florida", name: "Panthers" },
  LAK: { city: "Los Angeles", name: "Kings" },
  MIN: { city: "Minnesota", name: "Wild" },
  MTL: { city: "Montréal", name: "Canadiens" },
  NJD: { city: "New Jersey", name: "Devils" },
  NSH: { city: "Nashville", name: "Predators" },
  NYI: { city: "New York", name: "Islanders" },
  NYR: { city: "New York", name: "Rangers" },
  OTT: { city: "Ottawa", name: "Senators" },
  PHI: { city: "Philadelphia", name: "Flyers" },
  PIT: { city: "Pittsburgh", name: "Penguins" },
  SEA: { city: "Seattle", name: "Kraken" },
  SJS: { city: "San Jose", name: "Sharks" },
  STL: { city: "St. Louis", name: "Blues" },
  TBL: { city: "Tampa Bay", name: "Lightning" },
  TOR: { city: "Toronto", name: "Maple Leafs" },
  UTA: { city: "Utah", name: "Mammoth" },
  VAN: { city: "Vancouver", name: "Canucks" },
  VGK: { city: "Vegas", name: "Golden Knights" },
  WPG: { city: "Winnipeg", name: "Jets" },
  WSH: { city: "Washington", name: "Capitals" },
};

export const teamFullName = (ab: string): string => {
  const t = TEAM_INFO[ab];
  return t ? `${t.city} ${t.name}` : ab;
};

export const teamLogo = (ab: string): string => `https://assets.nhle.com/logos/nhl/svg/${ab}_light.svg`;
