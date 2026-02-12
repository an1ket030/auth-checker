// App.js — Premium / Vibrant Edition (Fixed & Polished)
import 'react-native-gesture-handler';
import React, { useState } from 'react';
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  Image, FlatList, StatusBar, Alert, SafeAreaView, ScrollView, ActivityIndicator, Dimensions
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import * as Haptics from 'expo-haptics';
import { MaterialCommunityIcons, Ionicons } from '@expo/vector-icons';

import { COLORS, FONTS, SIZES, SHADOWS } from './theme';
import { CleanCard, PrimaryButton, GradientCard } from './components/UI';
import { API_URL } from './config';

const { width } = Dimensions.get('window');

export default function App() {
  const [view, setView] = useState('login'); // login | home | result
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [result, setResult] = useState(null);

  // --- Actions ---
  const login = async (username, password) => {
    if (!username || !password) return Alert.alert("Missing Information", "Please enter both username and password.");
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/login`, { username, password });
      setUser(res.data);
      if (res.data.access_token) {
        axios.defaults.headers.common['Authorization'] = `Bearer ${res.data.access_token}`;
      }
      setView('home');
      fetchHistory(res.data.access_token);
    } catch (e) {
      console.log(e);
      Alert.alert("Login Failed", "Could not verify credentials. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const register = async (username, email, password) => {
    if (!username || !email || !password) return Alert.alert("Missing Information", "Please fill in all fields.");
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/register`, { username, email, password });
      setUser(res.data);
      if (res.data.access_token) {
        axios.defaults.headers.common['Authorization'] = `Bearer ${res.data.access_token}`;
      }
      setView('home');
      fetchHistory(res.data.access_token);
    } catch (e) {
      console.log(e);
      let msg = e.response?.data?.detail || "Could not create account.";
      if (Array.isArray(msg)) {
        msg = msg.map(err => err.msg).join('\n');
      } else if (typeof msg !== 'string') {
        msg = JSON.stringify(msg);
      }
      Alert.alert("Registration Failed", msg);
    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async (token) => {
    try {
      const res = await axios.get(`${API_URL}/history`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(res.data);
    } catch (e) { console.log(e); }
  };

  const pickImage = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.8,
    });
    if (!res.canceled) {
      startScan(res.assets[0]);
    }
  };

  const startScan = async (asset) => {
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", {
        uri: asset.uri,
        name: 'scan.jpg',
        type: 'image/jpeg'
      });

      const res = await axios.post(`${API_URL}/scan`, form, {
        headers: {
          'Content-Type': 'multipart/form-data',
          Authorization: `Bearer ${user.access_token}`
        }
      });

      setResult(res.data);
      Haptics.notificationAsync(
        res.data.label === 'AUTHENTIC'
          ? Haptics.NotificationFeedbackType.Success
          : Haptics.NotificationFeedbackType.Warning
      );
      setView('result');
      fetchHistory(user.access_token);
    } catch (e) {
      console.log("Scan Error:", e);
      let msg = e.response?.data?.detail || "Unable to process the image.";
      if (typeof msg !== 'string') msg = JSON.stringify(msg);
      Alert.alert("Scan Failed", msg);
    } finally {
      setLoading(false);
    }
  };

  // --- Views ---

  const LoginView = () => {
    const [isRegistering, setIsRegistering] = useState(false);
    const [u, setU] = useState('');
    const [e, setE] = useState('');
    const [p, setP] = useState('');

    const handleAuth = async () => {
      if (isRegistering) {
        await register(u, e, p);
      } else {
        await login(u, p);
      }
    }

    return (
      <View style={styles.centerContainer}>
        {/* Background Gradient */}
        <LinearGradient
          colors={['#E0E7FF', '#F3F4F6']}
          style={StyleSheet.absoluteFill}
        />

        <View style={{ width: '100%', maxWidth: 360 }}>
          <View style={{ marginBottom: 40, alignItems: 'center' }}>
            <View style={styles.logoCircle}>
              <MaterialCommunityIcons name="shield-check" size={48} color={COLORS.primary} />
            </View>
            <Text style={styles.title}>AuthChecker</Text>
            <Text style={styles.subtitle}>Secure Medicine Verification</Text>
          </View>

          <CleanCard style={{ padding: 30 }}>
            {isRegistering ? (
              <Text style={styles.cardHeader}>Create Account</Text>
            ) : (
              <Text style={styles.cardHeader}>Welcome Back</Text>
            )}

            <View style={styles.inputContainer}>
              <Ionicons name="person-outline" size={20} color={COLORS.textLight} style={{ marginRight: 10 }} />
              <TextInput
                style={styles.input}
                placeholder="Username"
                placeholderTextColor={COLORS.textLight}
                onChangeText={setU}
                autoCapitalize="none"
                value={u}
              />
            </View>

            {isRegistering && (
              <View style={styles.inputContainer}>
                <Ionicons name="mail-outline" size={20} color={COLORS.textLight} style={{ marginRight: 10 }} />
                <TextInput
                  style={styles.input}
                  placeholder="Email Address"
                  placeholderTextColor={COLORS.textLight}
                  onChangeText={setE}
                  keyboardType="email-address"
                  autoCapitalize="none"
                  value={e}
                />
              </View>
            )}

            <View style={styles.inputContainer}>
              <Ionicons name="lock-closed-outline" size={20} color={COLORS.textLight} style={{ marginRight: 10 }} />
              <TextInput
                style={styles.input}
                placeholder="Password"
                placeholderTextColor={COLORS.textLight}
                onChangeText={setP}
                secureTextEntry
                autoCapitalize="none"
                value={p}
              />
            </View>

            <View style={{ height: 20 }} />

            <PrimaryButton
              label={isRegistering ? "Sign Up" : "Sign In"}
              onPress={handleAuth}
              loading={loading}
            />
          </CleanCard>

          <TouchableOpacity
            onPress={() => setIsRegistering(!isRegistering)}
            style={{ marginTop: 20, alignItems: 'center', padding: 10 }}
          >
            <Text style={{ color: COLORS.textDim }}>
              {isRegistering ? "Already have an account? " : "Don't have an account? "}
              <Text style={{ color: COLORS.primary, fontWeight: '700' }}>
                {isRegistering ? "Login" : "Register"}
              </Text>
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  const LoadingOverlay = () => {
    if (!loading) return null;
    return (
      <View style={[StyleSheet.absoluteFill, styles.loadingOverlay]}>
        <CleanCard style={{ alignItems: 'center', padding: 40 }}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={{ marginTop: 20, fontSize: 16, fontWeight: '600', color: COLORS.text }}>Processing...</Text>
        </CleanCard>
      </View>
    )
  }

  const HomeView = () => (
    <View style={styles.container}>
      <LinearGradient
        colors={[COLORS.bg, '#E0E7FF']}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView style={{ flex: 1 }}>
        <View style={styles.header}>
          <View>
            <Text style={styles.headerSubtitle}>Hello, {user?.username} 👋</Text>
            <Text style={styles.headerTitle}>Dashboard</Text>
          </View>
          <TouchableOpacity style={styles.profileButton}>
            <Image
              source={{ uri: 'https://ui-avatars.com/api/?name=' + user?.username + '&background=random' }}
              style={{ width: 40, height: 40, borderRadius: 20 }}
            />
          </TouchableOpacity>
        </View>

        <ScrollView contentContainerStyle={{ padding: 20 }}>
          {/* Main Action - Gradient Card */}
          <GradientCard style={{ alignItems: 'center', paddingVertical: 40 }}>
            <TouchableOpacity onPress={pickImage} activeOpacity={0.8}>
              <View style={styles.scanIconWrapper}>
                <MaterialCommunityIcons name="barcode-scan" size={40} color={COLORS.primary} />
              </View>
            </TouchableOpacity>
            <Text style={styles.scanTextWhite}>Scan Medicine</Text>
            <Text style={styles.scanSubtextWhite}>Verify authenticity instantly</Text>
          </GradientCard>

          <Text style={styles.sectionHeader}>Recent Scans</Text>

          {history.map((item, i) => (
            <CleanCard key={i} style={styles.historyCard}>
              <View style={[styles.iconBox, { backgroundColor: item.status === 'AUTHENTIC' ? COLORS.successBg : COLORS.dangerBg }]}>
                <MaterialCommunityIcons
                  name={item.status === 'AUTHENTIC' ? "check-decagram" : "alert-decagram"}
                  size={24}
                  color={item.status === 'AUTHENTIC' ? COLORS.success : COLORS.danger}
                />
              </View>
              <View style={{ flex: 1, marginLeft: 15 }}>
                <Text style={styles.historyTitle} numberOfLines={1}>{item.product}</Text>
                <Text style={styles.historyDate}>{item.date}</Text>
              </View>
              <View style={{ alignItems: 'flex-end' }}>
                <Text style={[styles.historyScore, { color: item.status === 'AUTHENTIC' ? COLORS.success : COLORS.danger }]}>
                  {item.score}%
                </Text>
                <Text style={styles.historyStatus}>{item.status}</Text>
              </View>
            </CleanCard>
          ))}
          {history.length === 0 && <Text style={{ textAlign: 'center', color: COLORS.textDim, marginTop: 40, fontStyle: 'italic' }}>No scans yet. Start by tapping the blue card!</Text>}
        </ScrollView>
      </SafeAreaView>
    </View>
  );

  const ResultView = () => {
    if (!result) return null;

    // Fallback: use status if label is missing
    const statusText = result.label || result.status || "UNKNOWN";

    // Robust check: Authentic if label says so OR score is high
    const isAuthentic = statusText === 'AUTHENTIC' || (result.score >= 75);

    const displayLabel = isAuthentic ? "AUTHENTIC" : statusText;
    const statusColor = isAuthentic ? COLORS.success : COLORS.danger;
    const iconName = isAuthentic ? "shield-check" : "shield-alert";
    const breakdown = result.breakdown || {}; // Defensive fallback

    return (
      <View style={styles.container}>
        <LinearGradient
          colors={[isAuthentic ? '#ECFDF5' : '#FEF2F2', COLORS.bg]}
          style={StyleSheet.absoluteFill}
        />
        <SafeAreaView style={{ flex: 1 }}>
          <View style={styles.navBar}>
            <TouchableOpacity onPress={() => setView('home')} style={styles.navButton}>
              <Ionicons name="arrow-back" size={24} color={COLORS.text} />
            </TouchableOpacity>
            <Text style={styles.navTitle}>Scan Result</Text>
            <View style={{ width: 24 }} />
          </View>

          <ScrollView contentContainerStyle={{ padding: 20, alignItems: 'center' }}>

            <View style={[styles.resultCircle, { borderColor: statusColor, shadowColor: statusColor }]}>
              <MaterialCommunityIcons name={iconName} size={80} color={statusColor} />
            </View>

            <Text style={[styles.resultTitle, { color: statusColor }]}>{displayLabel}</Text>
            <Text style={styles.resultScore}>{result.score}% Trust Score</Text>

            <View style={{ width: '100%', marginTop: 30 }}>
              <Text style={styles.sectionHeader}>Verification Details</Text>
              <CleanCard>
                <InfoRow label="Product" value={result.product || "Unknown"} />
                <InfoRow label="Batch No." value={breakdown.batch_in_db ? "Verified in Database" : "Not Found"}
                  highlight={breakdown.batch_in_db} />
                <InfoRow label="Serial No." value={breakdown.serial_valid ? "Valid Serial" : "Invalid/Missing"}
                  highlight={breakdown.serial_valid} />
                <InfoRow label="Expiry Date" value={breakdown.mfg_exp_valid ? "Date Valid" : "Expired/Invalid"}
                  highlight={breakdown.mfg_exp_valid} />
                <InfoRow label="Packaging" value={`${Math.round(breakdown.packaging_sim * 100 || 0)}% Similarity`} />
              </CleanCard>
            </View>

            <PrimaryButton
              label="Scan Another Medicine"
              onPress={() => setView('home')}
              style={{ marginTop: 20 }}
            />
          </ScrollView>
        </SafeAreaView>
      </View>
    );
  };

  return (
    <>
      <StatusBar barStyle="dark-content" backgroundColor="#FFF" />
      {view === 'login' && <LoginView />}
      {view === 'home' && <HomeView />}
      {view === 'result' && <ResultView />}
      {loading && view !== 'login' && <LoadingOverlay />}
    </>
  );
}

const InfoRow = ({ label, value, highlight }) => (
  <View style={styles.infoRow}>
    <Text style={styles.infoLabel}>{label}</Text>
    <Text style={[
      styles.infoValue,
      highlight === true && { color: COLORS.success },
      highlight === false && { color: COLORS.danger }
    ]}>{value}</Text>
  </View>
);

const styles = StyleSheet.create({
  container: { flex: 1 },
  centerContainer: {
    flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20,
  },
  logoCircle: {
    width: 100, height: 100, borderRadius: 50,
    backgroundColor: '#FFF',
    alignItems: 'center', justifyContent: 'center', marginBottom: 20,
    ...SHADOWS.light
  },
  title: { fontSize: 32, fontWeight: '800', color: COLORS.text, marginBottom: 5 },
  subtitle: { fontSize: 16, color: COLORS.textDim },
  cardHeader: { fontSize: 22, fontWeight: '700', color: COLORS.text, marginBottom: 20, textAlign: 'center' },
  inputContainer: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: '#F9FAFB', borderRadius: 12, borderWidth: 1, borderColor: '#E5E7EB',
    paddingHorizontal: 15, marginBottom: 15, height: 55
  },
  input: { flex: 1, fontSize: 16, color: COLORS.text },

  loadingOverlay: {
    backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'center', alignItems: 'center', zIndex: 999
  },

  header: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 24, paddingVertical: 20,
  },
  headerTitle: { fontSize: 34, fontWeight: '800', color: COLORS.text },
  headerSubtitle: { fontSize: 16, color: COLORS.textDim, fontWeight: '500' },

  scanIconWrapper: {
    width: 80, height: 80, borderRadius: 40,
    backgroundColor: '#FFF', alignItems: 'center', justifyContent: 'center', marginBottom: 15,
    ...SHADOWS.medium
  },
  scanTextWhite: { fontSize: 22, fontWeight: '700', color: '#FFF' },
  scanSubtextWhite: { fontSize: 14, color: 'rgba(255,255,255,0.8)', marginTop: 5 },

  sectionHeader: { fontSize: 20, fontWeight: '700', marginTop: 10, marginBottom: 15, color: COLORS.text },

  historyCard: { flexDirection: 'row', alignItems: 'center', paddingVertical: 15 },
  iconBox: { width: 48, height: 48, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  historyTitle: { fontSize: 16, fontWeight: '700', color: COLORS.text },
  historyDate: { fontSize: 13, color: COLORS.textDim, marginTop: 2 },
  historyScore: { fontSize: 16, fontWeight: '700' },
  historyStatus: { fontSize: 11, color: COLORS.textDim, textTransform: 'uppercase', fontWeight: '600' },

  navBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingVertical: 15 },
  navTitle: { fontSize: 18, fontWeight: '700', color: COLORS.text },
  navButton: { padding: 5 },

  resultCircle: {
    width: 140, height: 140, borderRadius: 70, borderWidth: 6,
    backgroundColor: '#FFF', alignItems: 'center', justifyContent: 'center', marginBottom: 20,
    elevation: 10
  },
  resultTitle: { fontSize: 32, fontWeight: '800', marginBottom: 5 },
  resultScore: { fontSize: 18, fontWeight: '500', color: COLORS.textDim },

  infoRow: {
    flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 15,
    borderBottomWidth: 1, borderBottomColor: '#F3F4F6'
  },
  infoLabel: { fontSize: 16, color: COLORS.textDim, fontWeight: '500' },
  infoValue: { fontSize: 16, fontWeight: '600', color: COLORS.text }
});
