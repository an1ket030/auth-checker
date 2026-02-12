export const COLORS = {
    // Vibrant Gradient Colors
    primary: '#4F46E5',   // Indigo 600
    primaryDark: '#4338ca', // Indigo 700
    secondary: '#06B6D4', // Cyan 500
    accent: '#8B5CF6',    // Violet 500

    // Backgrounds
    bg: '#F3F4F6',        // Cool Gray 100
    suface: '#FFFFFF',

    // Status
    success: '#10B981',   // Emerald 500
    successBg: '#D1FAE5', // Emerald 100
    danger: '#EF4444',    // Red 500
    dangerBg: '#FEE2E2',  // Red 100
    warning: '#F59E0B',   // Amber 500

    // Text
    text: '#111827',      // Gray 900
    textDim: '#6B7280',   // Gray 500
    textLight: '#9CA3AF', // Gray 400
    white: '#FFFFFF',

    // Border
    border: '#E5E7EB',
};

export const SIZES = {
    padding: 24,
    radius: 16,
    h1: 32,
    h2: 24,
    h3: 18,
    body: 14,
};

export const SHADOWS = {
    light: {
        shadowColor: COLORS.primary,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 12,
        elevation: 4, // Android
    },
    medium: {
        shadowColor: COLORS.primary,
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.15,
        shadowRadius: 24,
        elevation: 8, // Android
    },
    dark: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.25,
        shadowRadius: 30,
        elevation: 10,
    }
};

export const FONTS = {
    // handled by default system fonts for now, but configured for weight
    bold: '700',
    semiBold: '600',
    medium: '500',
    regular: '400',
};
