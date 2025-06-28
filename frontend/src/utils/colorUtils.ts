/**
 * Color Utility Functions
 * Enterprise-grade color manipulation and theming utilities
 * Provides dynamic color adjustments based on primary color selection
 */

/**
 * Convert hex color to RGB object
 */
export function hexToRgbObject(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

/**
 * Convert hex color to RGB string for CSS use
 */
export function hexToRgb(hex: string): string {
  const cleanHex = hex.startsWith('#') ? hex : '#' + hex;
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(cleanHex);
  return result 
    ? `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}`
    : '78, 205, 196'; // Default teal RGB
}

/**
 * Convert RGB to hex color
 */
export function rgbToHex(r: number, g: number, b: number): string {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/**
 * Convert hex to HSL
 */
export function hexToHsl(hex: string): { h: number; s: number; l: number } | null {
  const rgb = hexToRgbObject(hex);
  if (!rgb) return null;
  
  const r = rgb.r / 255;
  const g = rgb.g / 255;
  const b = rgb.b / 255;
  
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0, s = 0, l = (max + min) / 2;
  
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      case b: h = ((r - g) / d + 4) / 6; break;
    }
  }
  
  return { h: h * 360, s: s * 100, l: l * 100 };
}

/**
 * Convert HSL to hex
 */
export function hslToHex(h: number, s: number, l: number): string {
  h = h / 360;
  s = s / 100;
  l = l / 100;
  
  let r, g, b;
  
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  
  return rgbToHex(Math.round(r * 255), Math.round(g * 255), Math.round(b * 255));
}

/**
 * Get the luminance of a color
 */
export function getLuminance(hex: string): number {
  const rgb = hexToRgbObject(hex);
  if (!rgb) return 0;
  
  const { r, g, b } = rgb;
  const [rs, gs, bs] = [r, g, b].map(c => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

/**
 * Get contrast ratio between two colors
 */
export function getContrastRatio(hex1: string, hex2: string): number {
  const lum1 = getLuminance(hex1);
  const lum2 = getLuminance(hex2);
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  return (brightest + 0.05) / (darkest + 0.05);
}

/**
 * Determine if text should be light or dark on a given background
 */
export function getContrastText(bgHex: string): string {
  const luminance = getLuminance(bgHex);
  return luminance > 0.5 ? '#000000' : '#ffffff';
}

/**
 * Adjust color lightness
 * @param hex - Hex color
 * @param amount - Amount to adjust (-100 to 100)
 */
export function adjustLightness(hex: string, amount: number): string {
  const hsl = hexToHsl(hex);
  if (!hsl) return hex;
  
  hsl.l = Math.max(0, Math.min(100, hsl.l + amount));
  return hslToHex(hsl.h, hsl.s, hsl.l);
}

/**
 * Adjust color saturation
 * @param hex - Hex color
 * @param amount - Amount to adjust (-100 to 100)
 */
export function adjustSaturation(hex: string, amount: number): string {
  const hsl = hexToHsl(hex);
  if (!hsl) return hex;
  
  hsl.s = Math.max(0, Math.min(100, hsl.s + amount));
  return hslToHex(hsl.h, hsl.s, hsl.l);
}

/**
 * Create color palette based on primary color
 */
export interface ColorPalette {
  primary: string;
  primaryDark: string;
  primaryLight: string;
  primaryVeryLight: string;
  primaryAlpha10: string;
  primaryAlpha20: string;
  primaryAlpha30: string;
  primaryAlpha50: string;
  
  // Semantic colors derived from primary
  success: string;
  successLight: string;
  warning: string;
  warningLight: string;
  error: string;
  errorLight: string;
  info: string;
  infoLight: string;
  
  // Greys for dark mode
  textPrimary: string;
  textSecondary: string;
  textTertiary: string;
  textDisabled: string;
  
  // Background colors
  bgPrimary: string;
  bgSecondary: string;
  bgTertiary: string;
  bgHover: string;
  
  // Border colors
  borderPrimary: string;
  borderSecondary: string;
  borderFocus: string;
}

/**
 * Generate a complete color palette from primary color
 */
export function generateColorPalette(primaryHex: string): ColorPalette {
  const rgb = hexToRgbObject(primaryHex);
  const hsl = hexToHsl(primaryHex);
  
  if (!rgb || !hsl) {
    throw new Error('Invalid color format');
  }
  
  // Primary variations
  const primaryDark = adjustLightness(primaryHex, -15);
  const primaryLight = adjustLightness(primaryHex, 15);
  const primaryVeryLight = adjustLightness(primaryHex, 30);
  
  // Alpha variations
  const primaryAlpha10 = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.1)`;
  const primaryAlpha20 = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.2)`;
  const primaryAlpha30 = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.3)`;
  const primaryAlpha50 = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.5)`;
  
  // Semantic colors - derived from primary hue
  const successHue = (hsl.h + 120) % 360; // Green shift
  const warningHue = (hsl.h + 60) % 360;  // Yellow shift
  const errorHue = 0; // Red
  const infoHue = 210; // Blue
  
  const success = hslToHex(successHue, 70, 50);
  const successLight = hslToHex(successHue, 70, 65);
  const warning = hslToHex(warningHue, 80, 50);
  const warningLight = hslToHex(warningHue, 80, 65);
  const error = hslToHex(errorHue, 75, 50);
  const errorLight = hslToHex(errorHue, 75, 65);
  const info = hslToHex(infoHue, 70, 50);
  const infoLight = hslToHex(infoHue, 70, 65);
  
  // Text colors for dark mode (lighter greys)
  const textPrimary = '#ffffff';
  const textSecondary = '#e0e0e0'; // Lighter than #999
  const textTertiary = '#b0b0b0';  // Lighter than #666
  const textDisabled = '#808080';   // Medium grey
  
  // Background colors
  const bgPrimary = '#0a0a0a';
  const bgSecondary = '#1a1a1a';
  const bgTertiary = '#2a2a2a';
  const bgHover = 'rgba(255, 255, 255, 0.08)';
  
  // Border colors
  const borderPrimary = 'rgba(255, 255, 255, 0.1)';
  const borderSecondary = 'rgba(255, 255, 255, 0.2)';
  const borderFocus = primaryHex;
  
  return {
    primary: primaryHex,
    primaryDark,
    primaryLight,
    primaryVeryLight,
    primaryAlpha10,
    primaryAlpha20,
    primaryAlpha30,
    primaryAlpha50,
    
    success,
    successLight,
    warning,
    warningLight,
    error,
    errorLight,
    info,
    infoLight,
    
    textPrimary,
    textSecondary,
    textTertiary,
    textDisabled,
    
    bgPrimary,
    bgSecondary,
    bgTertiary,
    bgHover,
    
    borderPrimary,
    borderSecondary,
    borderFocus
  };
}

/**
 * Mix two colors
 * @param color1 - First hex color
 * @param color2 - Second hex color
 * @param weight - Weight of first color (0-1)
 */
export function mixColors(color1: string, color2: string, weight: number = 0.5): string {
  const rgb1 = hexToRgbObject(color1);
  const rgb2 = hexToRgbObject(color2);
  
  if (!rgb1 || !rgb2) return color1;
  
  const r = Math.round(rgb1.r * weight + rgb2.r * (1 - weight));
  const g = Math.round(rgb1.g * weight + rgb2.g * (1 - weight));
  const b = Math.round(rgb1.b * weight + rgb2.b * (1 - weight));
  
  return rgbToHex(r, g, b);
}

/**
 * Get complementary color
 */
export function getComplementaryColor(hex: string): string {
  const hsl = hexToHsl(hex);
  if (!hsl) return hex;
  
  const complementaryHue = (hsl.h + 180) % 360;
  return hslToHex(complementaryHue, hsl.s, hsl.l);
}

/**
 * Get analogous colors
 */
export function getAnalogousColors(hex: string, count: number = 2): string[] {
  const hsl = hexToHsl(hex);
  if (!hsl) return [hex];
  
  const colors: string[] = [];
  const step = 30; // 30 degrees on color wheel
  
  for (let i = 1; i <= count; i++) {
    const hue1 = (hsl.h + step * i) % 360;
    const hue2 = (hsl.h - step * i + 360) % 360;
    colors.push(hslToHex(hue1, hsl.s, hsl.l));
    colors.push(hslToHex(hue2, hsl.s, hsl.l));
  }
  
  return colors.slice(0, count);
}

/**
 * Get triadic colors
 */
export function getTriadicColors(hex: string): string[] {
  const hsl = hexToHsl(hex);
  if (!hsl) return [hex];
  
  const hue1 = (hsl.h + 120) % 360;
  const hue2 = (hsl.h + 240) % 360;
  
  return [
    hslToHex(hue1, hsl.s, hsl.l),
    hslToHex(hue2, hsl.s, hsl.l)
  ];
}

/**
 * Adjust color brightness (compatible with legacy adjustColor)
 * @param hex - Hex color
 * @param percent - Percent to adjust (-100 to 100)
 */
export function adjustColor(hex: string, percent: number): string {
  try {
    const cleanHex = hex.replace('#', '');
    const num = parseInt(cleanHex, 16);
    
    if (isNaN(num)) {
      return hex;
    }
    
    const amt = Math.round(2.55 * percent);
    const R = (num >> 16) + amt;
    const G = (num >> 8 & 0x00FF) + amt;
    const B = (num & 0x0000FF) + amt;
    
    return '#' + (0x1000000 + 
      (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
      (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
      (B < 255 ? B < 1 ? 0 : B : 255))
      .toString(16)
      .slice(1);
  } catch (error) {
    return hex;
  }
}

/**
 * Apply CSS variables for theming
 */
export function applyThemeColors(primaryColor: string): void {
  const palette = generateColorPalette(primaryColor);
  const root = document.documentElement;
  
  // Apply all palette colors as CSS variables
  Object.entries(palette).forEach(([key, value]) => {
    const cssVarName = `--color-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`;
    root.style.setProperty(cssVarName, value);
  });
  
  // Also set RGB values for alpha variations
  const rgb = hexToRgbObject(primaryColor);
  if (rgb) {
    root.style.setProperty('--color-primary-rgb', `${rgb.r}, ${rgb.g}, ${rgb.b}`);
  }
}