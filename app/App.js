// App.js — Premium / Vibrant Edition (Fixed & Polished)
import 'react-native-gesture-handler';
import React, { useState } from 'react';
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity,
  Image, FlatList, StatusBar, Alert, SafeAreaView, ScrollView, ActivityIndicator, Dimensions, Animated
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import * as Haptics from 'expo-haptics';
import { MaterialCommunityIcons, Ionicons } from '@expo/vector-icons';

import { COLORS, FONTS, SIZES, SHADOWS } from './theme';
import { CleanCard, PrimaryButton, GradientCard, OutlineButton } from './components/UI';
import { API_URL } from './config';
import Swipeable from 'react-native-gesture-handler/Swipeable';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import Svg, { Circle } from 'react-native-svg';

const { width } = Dimensions.get('window');

// --- Standard Animation Wrappers ---
const FadeInView = ({ delay, children }) => {
  const fadeAnim = React.useRef(new Animated.Value(0)).current; // Initial value for opacity: 0
  const translateY = React.useRef(new Animated.Value(20)).current;

  React.useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        delay: delay,
        useNativeDriver: true,
      }),
      Animated.timing(translateY, {
        toValue: 0,
        duration: 500,
        delay: delay,
        useNativeDriver: true,
      })
    ]).start();
  }, [fadeAnim, translateY, delay]);

  return (
    <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY }] }}>
      {children}
    </Animated.View>
  );
};

export default function App() {
  const [view, setView] = useState('login'); // login | home | result
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [result, setResult] = useState(null);

  // --- Actions ---
  const deleteHistoryItem = (index) => {
    const updated = [...history];
    updated.splice(index, 1);
    setHistory(updated);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  };

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

  // --- Animations ---
  const TrustRing = ({ score, color }) => {
    const radius = 40;
    const strokeWidth = 8;
    const circumference = 2 * Math.PI * radius;

    // Simple count up animation using state
    const [displayScore, setDisplayScore] = useState(0);

    React.useEffect(() => {
      let start = 0;
      const end = score;
      if (start === end) return;

      const duration = 1000;
      const incrementTime = (duration / end) * 0.8;

      const timer = setInterval(() => {
        start += 1;
        setDisplayScore(start);
        if (start >= end) clearInterval(timer);
      }, incrementTime);

      return () => clearInterval(timer);
    }, [score]);

    return (
      <View style={{ alignItems: 'center', justifyContent: 'center' }}>
        <Svg width={100} height={100} viewBox="0 0 100 100">
          <Circle cx="50" cy="50" r={radius} stroke="#F1F5F9" strokeWidth={strokeWidth} fill="transparent" />
          {/* Static ring for stability, dynamic text */}
          <Circle
            cx="50" cy="50" r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="transparent"
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={circumference - (circumference * score) / 100}
            strokeLinecap="round"
            rotation="-90"
            origin="50, 50"
          />
        </Svg>
        <View style={StyleSheet.absoluteFillObject} justifyContent="center" alignItems="center">
          <Text style={{ fontSize: 20, fontWeight: '800', color: COLORS.text }}>{displayScore}%</Text>
        </View>
      </View>
    );
  };

  const CustomTabBar = () => {
    if (view === 'login' || view === 'result') return null;
    return (
      <View style={styles.tabBarContainer}>
        <View style={styles.tabBarGlass}>
          <TouchableOpacity onPress={() => setView('home')} style={styles.tabItem}>
            <Ionicons name={view === 'home' ? "home" : "home-outline"} size={24} color={view === 'home' ? COLORS.primary : COLORS.textDim} />
            <Text style={[styles.tabText, view === 'home' && { color: COLORS.primary }]}>Home</Text>
          </TouchableOpacity>

          <View style={{ width: 60 }} />

          <TouchableOpacity onPress={() => setView('profile')} style={styles.tabItem}>
            <Ionicons name={view === 'profile' ? "person" : "person-outline"} size={24} color={view === 'profile' ? COLORS.primary : COLORS.textDim} />
            <Text style={[styles.tabText, view === 'profile' && { color: COLORS.primary }]}>Profile</Text>
          </TouchableOpacity>
        </View>

        {/* Floating FAB */}
        <TouchableOpacity onPress={pickImage} style={styles.fabButton} activeOpacity={0.9}>
          <LinearGradient colors={[COLORS.primary, COLORS.accent]} style={styles.fabGradient}>
            <MaterialCommunityIcons name="barcode-scan" size={28} color="#FFF" />
          </LinearGradient>
        </TouchableOpacity>
      </View>
    );
  };

  // --- Screens ---

  const ProfileView = () => (
    <View style={styles.container}>
      <LinearGradient colors={['#F8FAFC', '#EFF6FF']} style={StyleSheet.absoluteFill} />
      <SafeAreaView style={{ flex: 1 }}>
        <View style={{ alignItems: 'center', paddingTop: 40, paddingBottom: 20 }}>
          <View style={[styles.profileAvatarBig, SHADOWS.medium]}>
            <Image
              source={{ uri: 'https://ui-avatars.com/api/?name=' + user?.username + '&background=0D9488&color=fff&size=200' }}
              style={{ width: 100, height: 100, borderRadius: 50 }}
            />
          </View>
          <Text style={styles.profileNameBig}>{user?.username}</Text>
          <Text style={styles.profileEmail}>{user?.email || "user@example.com"}</Text>
        </View>

        <ScrollView contentContainerStyle={{ padding: 20 }}>
          <View style={styles.statsGrid}>
            <CleanCard style={styles.statCard}>
              <Text style={styles.statNumber}>{history.length}</Text>
              <Text style={styles.statLabel}>Total Scans</Text>
            </CleanCard>
            <CleanCard style={styles.statCard}>
              <Text style={[styles.statNumber, { color: COLORS.success }]}>
                {history.filter(h => h.status === 'AUTHENTIC').length}
              </Text>
              <Text style={styles.statLabel}>Authentic</Text>
            </CleanCard>
          </View>

          <Text style={styles.sectionHeaderSmall}>Settings</Text>
          <TouchableOpacity style={styles.settingItem}>
            <Ionicons name="notifications-outline" size={22} color={COLORS.text} />
            <Text style={styles.settingText}>Notifications</Text>
            <Ionicons name="chevron-forward" size={20} color={COLORS.textDim} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.settingItem}>
            <Ionicons name="shield-checkmark-outline" size={22} color={COLORS.text} />
            <Text style={styles.settingText}>Security</Text>
            <Ionicons name="chevron-forward" size={20} color={COLORS.textDim} />
          </TouchableOpacity>

          <TouchableOpacity onPress={() => { setUser(null); setView('login'); }} style={[styles.settingItem, { marginTop: 20 }]}>
            <Ionicons name="log-out-outline" size={22} color={COLORS.danger} />
            <Text style={[styles.settingText, { color: COLORS.danger }]}>Log Out</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
      <CustomTabBar />
    </View>
  );

  const HomeView = () => (
    <View style={styles.container}>
      <LinearGradient colors={[COLORS.bg, '#F0F9FF']} style={StyleSheet.absoluteFill} />
      <SafeAreaView style={{ flex: 1 }}>
        <View style={[styles.header, { paddingBottom: 10 }]}>
          <View>
            <Text style={styles.headerSubtitle}>Hello, {user?.username} 👋</Text>
            <Text style={styles.headerTitle}>Dashboard</Text>
          </View>
          <TouchableOpacity onPress={() => setView('profile')}>
            <Image
              source={{ uri: 'https://ui-avatars.com/api/?name=' + user?.username + '&background=random' }}
              style={{ width: 40, height: 40, borderRadius: 20, borderWidth: 2, borderColor: '#FFF' }}
            />
          </TouchableOpacity>
        </View>

        <ScrollView contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 100, flexGrow: 1 }}>

          {/* Action Grid */}
          <View style={styles.actionGrid}>
            <TouchableOpacity onPress={pickImage} activeOpacity={0.8} style={[styles.actionButton, { backgroundColor: '#E0F2FE' }]}>
              <View style={[styles.actionIconCircle, { backgroundColor: '#0284C7' }]}>
                <Ionicons name="camera" size={28} color="#FFF" />
              </View>
              <Text style={[styles.actionLabel, { color: '#0284C7' }]}>Scan Label</Text>
              <Text style={styles.actionSub}>Use Camera</Text>
            </TouchableOpacity>

            <TouchableOpacity onPress={pickImage} activeOpacity={0.8} style={[styles.actionButton, { backgroundColor: '#DCFCE7' }]}>
              <View style={[styles.actionIconCircle, { backgroundColor: '#16A34A' }]}>
                <Ionicons name="images" size={28} color="#FFF" />
              </View>
              <Text style={[styles.actionLabel, { color: '#16A34A' }]}>Upload</Text>
              <Text style={styles.actionSub}>From Gallery</Text>
            </TouchableOpacity>
          </View>

          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 10, marginBottom: 15 }}>
            <Text style={styles.sectionHeader}>Recent Scans</Text>
          </View>

          {history.map((item, i) => (
            <FadeInView key={i} delay={i * 100}>
              <Swipeable
                renderRightActions={() => (
                  <TouchableOpacity onPress={() => deleteHistoryItem(i)} style={styles.deleteAction}>
                    <Ionicons name="trash-outline" size={24} color="#FFF" />
                  </TouchableOpacity>
                )}
              >
                <View style={styles.historyCardModern}>
                  <View style={[styles.statusStrip, { backgroundColor: item.status === 'AUTHENTIC' ? COLORS.success : COLORS.danger }]} />
                  <View style={{ padding: 15, flexDirection: 'row', alignItems: 'center', flex: 1 }}>
                    <View style={[styles.iconBoxSmall, { backgroundColor: item.status === 'AUTHENTIC' ? '#DCFCE7' : '#FEE2E2' }]}>
                      <MaterialCommunityIcons name={item.status === 'AUTHENTIC' ? "check" : "alert-circle-outline"} size={20} color={item.status === 'AUTHENTIC' ? COLORS.success : COLORS.danger} />
                    </View>
                    <View style={{ marginLeft: 12, flex: 1 }}>
                      <Text style={styles.historyTitle} numberOfLines={1}>{item.product}</Text>
                      <Text style={styles.historyDate}>{item.date}</Text>
                    </View>
                    <View style={[styles.statusPill, { backgroundColor: item.status === 'AUTHENTIC' ? '#DCFCE7' : '#FEE2E2' }]}>
                      <Text style={[styles.statusPillText, { color: item.status === 'AUTHENTIC' ? '#166534' : '#991B1B' }]}>{item.status}</Text>
                    </View>
                  </View>
                </View>
              </Swipeable>
            </FadeInView>
          ))}
          {history.length === 0 && (
            <View style={{ alignItems: 'center', marginTop: 40, opacity: 0.5 }}>
              <MaterialCommunityIcons name="history" size={40} color={COLORS.textDim} />
              <Text style={{ marginTop: 10, color: COLORS.textDim }}>No scans yet</Text>
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
      <CustomTabBar />
    </View>
  );

  const ResultView = () => {
    if (!result) return null;
    const statusText = result.label || result.status || "UNKNOWN";
    const isAuthentic = statusText === 'AUTHENTIC' || (result.score >= 75);
    const displayLabel = isAuthentic ? "AUTHENTIC" : statusText;
    const statusColor = isAuthentic ? COLORS.success : COLORS.danger;
    const breakdown = result.breakdown || {};

    return (
      <View style={styles.container}>
        <View style={[styles.resultHeader, { backgroundColor: statusColor }]}>
          <SafeAreaView>
            <View style={styles.resultNavBar}>
              <TouchableOpacity onPress={() => setView('home')} style={styles.navButtonWhite}>
                <Ionicons name="arrow-back" size={24} color="#FFF" />
              </TouchableOpacity>
              <Text style={styles.navTitleWhite}>Verification Result</Text>
              <TouchableOpacity onPress={() => { }}><Ionicons name="share-social-outline" size={24} color="#FFF" /></TouchableOpacity>
            </View>
            <View style={{ alignItems: 'center', marginTop: 10, marginBottom: 30 }}>
              <MaterialCommunityIcons name="shield-check" size={64} color="#FFF" />
            </View>
          </SafeAreaView>
        </View>

        <ScrollView contentContainerStyle={{ padding: 20, paddingTop: 0, flexGrow: 1 }} style={{ marginTop: -20 }}>
          <View style={styles.resultMainCard}>
            <View style={[styles.statusPillLarge, { backgroundColor: isAuthentic ? '#DCFCE7' : '#FEE2E2' }]}>
              <MaterialCommunityIcons name={isAuthentic ? "check-circle" : "alert"} size={16} color={statusColor} />
              <Text style={[styles.statusPillTextLarge, { color: statusColor }]}>{displayLabel}</Text>
            </View>

            <View style={{ flexDirection: 'row', alignItems: 'center', marginVertical: 20 }}>
              <View style={{ marginRight: 20 }}>
                <TrustRing score={result.score} color={statusColor} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.detectedTitle}>Detected product</Text>
                <Text style={styles.detectedName}>{result.product || "Unknown"}</Text>
              </View>
            </View>

            <View style={styles.divider} />
            <Text style={styles.analysisTitle}>Analysis Summary</Text>
            <Text style={styles.analysisText}>{result.reason || "AI analysis completed."}</Text>
            <View style={{ height: 15 }} />
            <InfoRow label="Batch No." value={breakdown.batch_in_db ? "Verified" : "Not Found"} highlight={breakdown.batch_in_db} />
            <InfoRow label="Serial No." value={breakdown.serial_valid ? "Valid" : "Invalid"} highlight={breakdown.serial_valid} />
          </View>
          <PrimaryButton label="Scan Another" onPress={() => setView('home')} />
        </ScrollView>
      </View>
    );
  };

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
              <MaterialCommunityIcons name="shield-check" size={56} color={COLORS.primary} />
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

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFF" />
      {view === 'login' && <LoginView />}
      {view === 'home' && <HomeView />}
      {view === 'profile' && <ProfileView />}
      {view === 'result' && <ResultView />}
      {loading && view !== 'login' && <LoadingOverlay />}
    </GestureHandlerRootView>
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
  infoLabel: { fontSize: 14, color: COLORS.textDim, fontWeight: '500' },
  infoValue: { fontSize: 14, fontWeight: '600', color: COLORS.text },

  // --- TAB BAR ---
  tabBarContainer: {
    position: 'absolute', bottom: 30, left: 20, right: 20, height: 70,
    alignItems: 'center', justifyContent: 'center'
  },
  tabBarGlass: {
    flexDirection: 'row', backgroundColor: '#FFF', borderRadius: 35,
    height: 70, width: '100%', alignItems: 'center', justifyContent: 'space-around',
    shadowColor: "#000", shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.1, shadowRadius: 20, elevation: 10,
    paddingHorizontal: 20
  },
  tabItem: { alignItems: 'center', justifyContent: 'center' },
  tabText: { fontSize: 10, fontWeight: '600', marginTop: 4, color: COLORS.textDim },

  fabButton: {
    position: 'absolute', top: -25, width: 70, height: 70,
    borderRadius: 35, ...SHADOWS.medium,
    justifyContent: 'center', alignItems: 'center',
    backgroundColor: '#FFF', padding: 4
  },
  fabGradient: {
    width: '100%', height: '100%', borderRadius: 35,
    alignItems: 'center', justifyContent: 'center'
  },

  // --- PROFILE VIEW ---
  profileAvatarBig: {
    padding: 4, backgroundColor: '#FFF', borderRadius: 52, marginBottom: 15
  },
  profileNameBig: { fontSize: 24, fontWeight: '800', color: COLORS.text },
  profileEmail: { fontSize: 14, color: COLORS.textDim, marginBottom: 30 },

  statsGrid: { flexDirection: 'row', gap: 15, marginBottom: 30 },
  statCard: { flex: 1, padding: 20, alignItems: 'center', justifyContent: 'center' },
  statNumber: { fontSize: 28, fontWeight: '800', color: COLORS.primary },
  statLabel: { fontSize: 12, color: COLORS.textDim, textTransform: 'uppercase', marginTop: 5 },

  settingItem: {
    flexDirection: 'row', alignItems: 'center', paddingVertical: 15,
    borderBottomWidth: 1, borderBottomColor: '#F1F5F9'
  },
  settingText: { flex: 1, marginLeft: 15, fontSize: 16, fontWeight: '600', color: COLORS.text },

  // --- SHARED ---
  actionGrid: { flexDirection: 'row', gap: 15, marginBottom: 20 },
  actionButton: {
    flex: 1, padding: 16, borderRadius: 16, alignItems: 'center', justifyContent: 'center',
    ...SHADOWS.light
  },
  actionIconCircle: { width: 50, height: 50, borderRadius: 25, alignItems: 'center', justifyContent: 'center', marginBottom: 10 },
  actionLabel: { fontSize: 15, fontWeight: '700', marginBottom: 2 },
  actionSub: { fontSize: 12, color: COLORS.textDim },

  badge: { backgroundColor: COLORS.primary, borderRadius: 12, paddingHorizontal: 8, paddingVertical: 2 },
  badgeText: { color: '#FFF', fontSize: 12, fontWeight: '700' },

  historyCardModern: {
    backgroundColor: '#FFF', borderRadius: 12, flexDirection: 'row', marginBottom: 0,
    shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 2,
    overflow: 'hidden'
  },
  statusStrip: { width: 6, height: '100%' },
  iconBoxSmall: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
  statusPill: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8 },
  statusPillText: { fontSize: 10, fontWeight: '700' },
  deleteAction: {
    backgroundColor: COLORS.danger, justifyContent: 'center', alignItems: 'center', width: 80,
    marginBottom: 0, borderRadius: 12, marginLeft: 10, height: '100%'
  },

  // Reuse existing result styles
  resultHeader: { paddingBottom: 30, borderBottomLeftRadius: 30, borderBottomRightRadius: 30 },
  resultNavBar: { flexDirection: 'row', justifyContent: 'space-between', padding: 20, alignItems: 'center' },
  navButtonWhite: { padding: 5, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 8 },
  navTitleWhite: { fontSize: 18, fontWeight: '700', color: '#FFF' },
  resultMainCard: {
    backgroundColor: '#FFF', borderRadius: 24, padding: 24,
    ...SHADOWS.medium, marginBottom: 20
  },
  statusPillLarge: { alignSelf: 'center', flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20, gap: 6 },
  statusPillTextLarge: { fontSize: 14, fontWeight: '700' },
  detectedTitle: { fontSize: 12, color: COLORS.textDim, textTransform: 'uppercase', fontWeight: '600', marginBottom: 4 },
  detectedName: { fontSize: 18, fontWeight: '700', color: COLORS.text },
  divider: { height: 1, backgroundColor: '#F1F5F9', marginVertical: 15 },
  analysisTitle: { fontSize: 16, fontWeight: '700', color: COLORS.text, marginBottom: 8 },
  analysisText: { fontSize: 14, color: COLORS.textDim, lineHeight: 22 },
  sectionHeader: { fontSize: 20, fontWeight: '700', marginTop: 10, marginBottom: 15, color: COLORS.text },
  sectionHeaderSmall: { fontSize: 16, fontWeight: '700', color: COLORS.text, marginBottom: 12 },
  scannedImageContainer: {
    width: '100%', height: 200, backgroundColor: '#F1F5F9', borderRadius: 16,
    alignItems: 'center', justifyContent: 'center', marginBottom: 20
  }
});
