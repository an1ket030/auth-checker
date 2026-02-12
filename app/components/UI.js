import React from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { COLORS, SIZES, SHADOWS, FONTS } from '../theme';

export const CleanCard = ({ children, style }) => {
    return (
        <View style={[styles.cardContainer, style]}>
            {children}
        </View>
    );
};

export const GradientCard = ({ children, style }) => {
    return (
        <LinearGradient
            colors={[COLORS.primary, COLORS.accent]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={[styles.gradientCard, style]}
        >
            {children}
        </LinearGradient>
    );
};

export const PrimaryButton = ({ label, onPress, loading, style }) => {
    return (
        <TouchableOpacity
            activeOpacity={0.8}
            onPress={onPress}
            disabled={loading}
            style={[styles.buttonWrapper, style]}
        >
            <LinearGradient
                colors={[COLORS.primary, COLORS.accent]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.gradientButton}
            >
                {loading ? (
                    <ActivityIndicator color="#FFF" />
                ) : (
                    <Text style={styles.buttonText}>{label}</Text>
                )}
            </LinearGradient>
        </TouchableOpacity>
    );
};

export const OutlineButton = ({ label, onPress, style }) => {
    return (
        <TouchableOpacity onPress={onPress} style={[styles.outlineButton, style]}>
            <Text style={styles.outlineButtonText}>{label}</Text>
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
    cardContainer: {
        backgroundColor: COLORS.suface,
        borderRadius: SIZES.radius,
        padding: SIZES.padding,
        ...SHADOWS.light,
        marginBottom: 16,
    },
    gradientCard: {
        borderRadius: SIZES.radius,
        padding: SIZES.padding,
        marginBottom: 16,
        ...SHADOWS.medium,
    },
    buttonWrapper: {
        width: '100%',
        borderRadius: 12,
        marginBottom: 12,
        ...SHADOWS.medium,
    },
    gradientButton: {
        paddingVertical: 16,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
    },
    buttonText: {
        color: '#FFF',
        fontSize: 16,
        fontWeight: FONTS.bold,
        letterSpacing: 0.5,
    },
    outlineButton: {
        paddingVertical: 15,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: COLORS.primary,
        alignItems: 'center',
        marginBottom: 12,
    },
    outlineButtonText: {
        color: COLORS.primary,
        fontSize: 16,
        fontWeight: FONTS.semiBold,
    }
});
