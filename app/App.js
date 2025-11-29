// App.js — updated: medical-blue header card behind "Welcome, username"
// NOTE: uses local HEADER_IMAGE path you uploaded: /mnt/data/4c3be9d0-bac7-468b-9254-6b118660d039.png

import { MaterialCommunityIcons } from "@expo/vector-icons";
import axios from "axios";
import * as ImagePicker from "expo-image-picker";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Animated,
  Dimensions,
  FlatList,
  Image,
  ImageBackground,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  UIManager,
  View,
} from "react-native";
import 'react-native-gesture-handler'; // must be first if using gesture-handler
import { GestureHandlerRootView, Swipeable } from "react-native-gesture-handler";

if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

const { width } = Dimensions.get("window");

// Use the local image you uploaded (bundler/tooling will transform path when packaging)
const HEADER_IMAGE = "file:///mnt/data/4c3be9d0-bac7-468b-9254-6b118660d039.png";
const API_URL = "http://10.154.202.52:8000/api/v1";

const THEME = {
  primary: "#075B9A",
  primaryLight: "#0D7AB3",
  accent: "#00A3B4",
  success: "#1EA08A",
  danger: "#E74C3C",
  bg: "#F4F7FB",
  surface: "#FFFFFF",
  text: "#0E2640",
  textLight: "#5B7088",
  muted: "#7F8A99",
  subtle: "rgba(7,91,154,0.06)",
  subtleAlt: "rgba(13,122,185,0.06)",
  border: "#E6EEF8",
  shadow: "rgba(0,0,0,0.08)",
};

const IconButton = ({ name, size = 20, color = THEME.primary, onPress, style }) => (
  <TouchableOpacity onPress={onPress} style={[styles.iconBtn, style]} activeOpacity={0.8}>
    <MaterialCommunityIcons name={name} size={size} color={color} />
  </TouchableOpacity>
);

const PrimaryButton = ({ label, icon, onPress, loading = false, secondary = false, style }) => (
  <TouchableOpacity
    activeOpacity={0.9}
    onPress={onPress}
    style={[styles.primaryBtn, secondary && styles.secondaryBtn, loading && { opacity: 0.7 }, style]}
    disabled={loading}
  >
    <View style={styles.btnInner}>
      <MaterialCommunityIcons name={icon} size={18} color={secondary ? THEME.primary : THEME.surface} style={{ marginRight: 10 }} />
      <Text style={[styles.btnText, secondary && { color: THEME.primary }]} numberOfLines={1} ellipsizeMode="tail">
        {label}
      </Text>
    </View>
    {loading && <ActivityIndicator style={{ marginLeft: 8 }} color={secondary ? THEME.primary : THEME.surface} />}
  </TouchableOpacity>
);

const Card = ({ children, style }) => <View style={[styles.card, style]}>{children}</View>;

export default function App() {
  const [view, setView] = useState("login"); // 'login' | 'home' | 'result'
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loginLoading, setLoginLoading] = useState(false);
  const [imageAsset, setImageAsset] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  // Undo state
  const [undoVisible, setUndoVisible] = useState(false);
  const undoTimerRef = useRef(null);
  const lastDeletedRef = useRef(null); // { item, index }

  // uncontrolled inputs to avoid focus loss
  const usernameRef = useRef("");
  const emailRef = useRef("");

  const scale = useRef(new Animated.Value(0)).current;
  const fade = useRef(new Animated.Value(0)).current;
  const pulse = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (view === "result" && result) {
      Animated.parallel([
        Animated.spring(scale, { toValue: 1, friction: 9, useNativeDriver: true }),
        Animated.timing(fade, { toValue: 1, duration: 300, useNativeDriver: true }),
      ]).start();

      Animated.loop(
        Animated.sequence([
          Animated.timing(pulse, { toValue: 1.04, duration: 900, useNativeDriver: true }),
          Animated.timing(pulse, { toValue: 1, duration: 900, useNativeDriver: true }),
        ])
      ).start();
    } else {
      scale.setValue(0);
      fade.setValue(0);
    }
    return () => {
      if (undoTimerRef.current) clearTimeout(undoTimerRef.current);
    };
  }, [view, result]);

  function isValidEmail(s) {
    if (!s || typeof s !== "string") return false;
    const t = s.trim().toLowerCase();
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(t);
  }

  const login = async () => {
    const username = (usernameRef.current || "").trim();
    const email = (emailRef.current || "").trim();

    if (!username) {
      Alert.alert("Missing Info", "Please enter your username.");
      return;
    }
    if (!email) {
      Alert.alert("Missing Info", "Please enter your email address.");
      return;
    }
    if (!isValidEmail(email)) {
      Alert.alert("Invalid Email", "Please enter a valid email like example@gmail.com.");
      return;
    }

    setLoginLoading(true);
    try {
      const res = await axios.post(`${API_URL}/login`, { username, email }, { timeout: 7000 });
      setUser(res.data);
      setView("home");
      fetchHistory(res.data.id);
    } catch (err) {
      console.warn("Login error", err?.message || err);
      Alert.alert("Login failed", "Cannot reach server. Check API URL and network.");
    } finally {
      setLoginLoading(false);
    }
  };

  const fetchHistory = async (userId) => {
    try {
      const res = await axios.get(`${API_URL}/history/${userId}`);
      setHistory(res.data || []);
    } catch (err) {
      console.warn("History fetch error", err?.message || err);
    }
  };

  const pickImage = async (useCamera = true) => {
    try {
      const perm = useCamera ? await ImagePicker.requestCameraPermissionsAsync() : await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert("Permission required", "Allow camera/gallery access.");
        return;
      }
      const picked = useCamera
        ? await ImagePicker.launchCameraAsync({ quality: 0.8, allowsEditing: true, aspect: [4, 3] })
        : await ImagePicker.launchImageLibraryAsync({ quality: 0.8, allowsEditing: true, aspect: [4, 3] });

      if (!picked.canceled) {
        const asset = picked.assets?.[0] || picked;
        setImageAsset(asset);
        await analyzeImage(asset);
      }
    } catch (err) {
      console.warn("Pick error", err);
      Alert.alert("Error", "Unable to open camera/gallery.");
    }
  };

  const analyzeImage = async (asset) => {
    if (!user) {
      Alert.alert("Not signed in", "Please sign in to scan.");
      return;
    }
    setLoading(true);
    try {
      const form = new FormData();
      const uri = asset.uri;
      const filename = uri.split("/").pop();
      const match = /\.(\w+)$/.exec(filename || "");
      const type = match ? `image/${match[1].toLowerCase()}` : "image/jpeg";
      form.append("file", { uri, name: filename || "scan.jpg", type });
      const res = await axios.post(`${API_URL}/scan?user_id=${user.id}`, form, { headers: { "Content-Type": "multipart/form-data" }, timeout: 60000 });
      setResult(res.data);
      setView("result");
      fetchHistory(user.id);
    } catch (err) {
      console.warn("Analyze error", err?.response || err?.message || err);
      const msg = err?.response?.data?.detail || "Verification failed. Try another photo.";
      Alert.alert("Scan failed", String(msg));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAtIndex = (index) => {
    const item = history[index];
    lastDeletedRef.current = { item, index };

    setHistory((prev) => {
      const copy = [...prev];
      copy.splice(index, 1);
      return copy;
    });

    setUndoVisible(true);
    if (undoTimerRef.current) clearTimeout(undoTimerRef.current);

    undoTimerRef.current = setTimeout(() => {
      setUndoVisible(false);
      lastDeletedRef.current = null;
      undoTimerRef.current = null;
      // optionally call backend delete endpoint here if available
    }, 5000);
  };

  const handleUndo = () => {
    const saved = lastDeletedRef.current;
    if (!saved) return;
    setHistory((prev) => {
      const copy = [...prev];
      const idx = Math.max(0, Math.min(saved.index, copy.length));
      copy.splice(idx, 0, saved.item);
      return copy;
    });
    setUndoVisible(false);
    if (undoTimerRef.current) { clearTimeout(undoTimerRef.current); undoTimerRef.current = null; }
    lastDeletedRef.current = null;
  };

  const renderRightActions = (progress, dragX, onPress) => {
    const trans = dragX.interpolate({ inputRange: [-100, 0], outputRange: [0, 100], extrapolate: "clamp" });
    return (
      <Animated.View style={{ transform: [{ translateX: trans }] }}>
        <TouchableOpacity onPress={onPress} activeOpacity={0.8} style={styles.deleteAction}>
          <MaterialCommunityIcons name="trash-can" size={22} color="#fff" />
          <Text style={styles.deleteText}>Delete</Text>
        </TouchableOpacity>
      </Animated.View>
    );
  };

  const HistoryRow = ({ item, index }) => {
    const ok = item.status === "AUTHENTIC";
    return (
      <GestureHandlerRootView>
        <Swipeable
          friction={2}
          rightThreshold={40}
          renderRightActions={(progress, dragX) =>
            renderRightActions(progress, dragX, () => {
              handleDeleteAtIndex(index);
            })
          }
        >
          <View style={[styles.historyRow, { borderLeftColor: ok ? THEME.success : THEME.danger }]}>
            <View style={[styles.historyIcon, { backgroundColor: ok ? THEME.subtleAlt : "rgba(231,76,60,0.06)" }]}>
              <MaterialCommunityIcons name={ok ? "check-circle" : "alert-circle"} size={18} color={ok ? THEME.success : THEME.danger} />
            </View>
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={styles.historyTitle}>{item.product}</Text>
              <Text style={styles.historyMeta}>{item.date} • {Math.round(item.score)}%</Text>
            </View>
            <View style={[styles.pill, { backgroundColor: ok ? "#E9FAF4" : "#FFF2F2" }]}>
              <Text style={[styles.pillText, { color: ok ? THEME.success : THEME.danger }]}>{item.status}</Text>
            </View>
          </View>
        </Swipeable>
      </GestureHandlerRootView>
    );
  };

  const ResultCard = ({ result }) => {
    const ok = (result?.score ?? 0) >= 60;
    return (
      <Animated.View style={[styles.resultCard, { transform: [{ scale }], opacity: fade }]}>
        <View style={[styles.resultTop, { backgroundColor: ok ? THEME.success : THEME.danger }]}>
          <Animated.View style={{ transform: [{ scale: pulse }] }}>
            <MaterialCommunityIcons name={ok ? "shield-check" : "shield-alert"} size={56} color={THEME.surface} />
          </Animated.View>
        </View>

        <View style={styles.resultContent}>
          <View style={[styles.statusBadge, { backgroundColor: ok ? "#E9FAF4" : "#FFF2F2", marginBottom: 14 }]}>
            <MaterialCommunityIcons name={ok ? "check" : "alert"} size={16} color={ok ? THEME.success : THEME.danger} style={{ marginRight: 6 }} />
            <Text style={[styles.statusBadgeText, { color: ok ? THEME.success : THEME.danger }]}>{ok ? "AUTHENTIC" : "SUSPICIOUS"}</Text>
          </View>

          <View style={[styles.scoreSection, { marginBottom: 8 }]}>
            <View style={[styles.scoreCircle, { backgroundColor: ok ? THEME.subtleAlt : "rgba(231,76,60,0.06)" }]}>
              <Text style={styles.scoreNumber}>{Math.round(result?.score ?? 0)}</Text>
              <Text style={styles.scorePercent}>%</Text>
            </View>
            <View style={{ marginLeft: 16, justifyContent: "center", flex: 1 }}>
              <Text style={styles.scoreLabel}>Trust Score</Text>
              <Text style={styles.scoreSubtext}>Overall confidence</Text>
            </View>
          </View>

          <View style={styles.progressContainer}>
            <View style={styles.progressBg}>
              <View style={[styles.progressFill, { width: `${result?.score ?? 0}%`, backgroundColor: ok ? THEME.success : THEME.danger }]} />
            </View>
          </View>

          <View style={{ marginTop: 12 }}>
            <Text style={{ fontWeight: "800", color: THEME.text }}>Detected product</Text>
            <Text style={{ color: THEME.textLight, marginTop: 6 }}>{result?.product ?? "Unknown"}</Text>
          </View>

          <View style={{ marginTop: 12 }}>
            <Text style={{ fontWeight: "800", color: THEME.text }}>Analysis Summary</Text>
            <Text style={{ color: THEME.textLight, marginTop: 6 }}>
              {result?.reason ? (result.reason.length > 220 ? result.reason.slice(0, 220) + "..." : result.reason) : "No details available."}
            </Text>
          </View>
        </View>
      </Animated.View>
    );
  };

  // SCREENS
  const LoginScreen = () => (
    <SafeAreaView style={[styles.safe, { paddingTop: Platform.OS === "android" ? StatusBar.currentHeight || 12 : 0 }]}>
      <StatusBar barStyle="light-content" />
      <ImageBackground source={{ uri: HEADER_IMAGE }} style={styles.header} imageStyle={{ opacity: 0.94 }}>
        <View style={styles.headerOverlay} />
        <View style={styles.headerContent}>
          <View style={styles.logoWrap}>
            <MaterialCommunityIcons name="hospital-box" size={36} color={THEME.surface} />
          </View>
          <Text style={styles.appTitle}>AuthChecker Pro</Text>
          <Text style={styles.appSubtitle}>Enterprise Medicine Verification</Text>
        </View>
      </ImageBackground>

      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={styles.loginContent} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="always">
          <Card>
            <Text style={styles.cardTitle}>Welcome Back</Text>
            <Text style={styles.cardSubtitle}>Sign in to your account</Text>

            <View style={styles.fieldsContainer}>
              <View style={styles.field}>
                <MaterialCommunityIcons name="account" size={18} color={THEME.primary} />
                <TextInput
                  style={styles.input}
                  placeholder="Username"
                  placeholderTextColor={THEME.muted}
                  onChangeText={(t) => (usernameRef.current = t)}
                  autoCapitalize="none"
                  autoCorrect={false}
                  returnKeyType="next"
                />
              </View>

              <View style={styles.field}>
                <MaterialCommunityIcons name="email" size={18} color={THEME.primary} />
                <TextInput
                  style={styles.input}
                  placeholder="Email Address (example@gmail.com)"
                  placeholderTextColor={THEME.muted}
                  onChangeText={(t) => (emailRef.current = t)}
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="email-address"
                  returnKeyType="done"
                />
              </View>
            </View>

            <PrimaryButton label={loginLoading ? "Signing in..." : "Sign In"} icon="login" onPress={login} loading={loginLoading} />
            <Text style={styles.disclaimer}>Healthcare professionals only. Scans are encrypted and secure.</Text>
          </Card>

          <Text style={styles.version}>v2.0 • Aniket Singh</Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );

  // HOME: medical-blue card behind welcome area
  const HomeScreen = () => (
    <SafeAreaView style={[styles.safe, { paddingTop: Platform.OS === "android" ? StatusBar.currentHeight || 12 : 0 }]}>
      <StatusBar barStyle="dark-content" />

      {/* This is the new medical-blue header card (rounded) placed over the page */}
      <View style={styles.homeHeaderContainer}>
        <ImageBackground source={{ uri: HEADER_IMAGE }} style={styles.homeHeaderBg} imageStyle={{ borderRadius: 18, opacity: 0.95 }}>
          <View style={styles.homeHeaderOverlay} />
          <View style={styles.homeHeaderInner}>
            <View style={styles.homeHeaderLeft}>
              <View style={styles.medicalCard}>
                <Text style={styles.welcomeInCard}>Welcome,</Text>
                <Text style={styles.usernameInCard}>{user?.username || ""}</Text>
                <Text style={styles.welcomeSubInCard}>Verify medicines instantly</Text>
              </View>
            </View>

            <View style={styles.homeHeaderRight}>
              <IconButton name="logout" size={22} onPress={() => { setUser(null); setView("login"); }} style={styles.logoutBtn} />
            </View>
          </View>
        </ImageBackground>
      </View>

      <ScrollView contentContainerStyle={{ paddingBottom: 40 }} showsVerticalScrollIndicator={false}>
        <View style={styles.actionGrid}>
          <TouchableOpacity style={[styles.actionCard, { backgroundColor: "#F8FBFF" }]} onPress={() => pickImage(true)} activeOpacity={0.85}>
            <View style={[styles.actionIcon, { backgroundColor: "rgba(7,91,154,0.08)" }]}>
              <MaterialCommunityIcons name="camera" size={32} color={THEME.primary} />
            </View>
            <Text style={styles.actionCardTitle}>Scan Label</Text>
            <Text style={styles.actionCardSub}>Use camera</Text>
          </TouchableOpacity>

          <TouchableOpacity style={[styles.actionCard, { backgroundColor: "#F7FFFE" }]} onPress={() => pickImage(false)} activeOpacity={0.85}>
            <View style={[styles.actionIcon, { backgroundColor: "rgba(0,163,180,0.06)" }]}>
              <MaterialCommunityIcons name="image-multiple" size={32} color={THEME.accent} />
            </View>
            <Text style={styles.actionCardTitle}>Upload Image</Text>
            <Text style={styles.actionCardSub}>From gallery</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Recent Verifications</Text>
            {history.length > 0 && (
              <View style={styles.historyBadge}>
                <Text style={styles.historyBadgeText}>{history.length}</Text>
              </View>
            )}
          </View>

          {history.length === 0 ? (
            <View style={styles.emptyState}>
              <MaterialCommunityIcons name="magnify" size={48} color={THEME.border} />
              <Text style={styles.emptyTitle}>No Verifications Yet</Text>
              <Text style={styles.emptySub}>Start by scanning a medicine label above.</Text>
            </View>
          ) : (
            <FlatList
              data={history}
              keyExtractor={(item, idx) => idx.toString()}
              renderItem={({ item, index }) => <HistoryRow item={item} index={index} />}
              scrollEnabled={false}
            />
          )}
        </View>
      </ScrollView>

      {undoVisible && (
        <View style={styles.undoBar}>
          <Text style={styles.undoText}>Item deleted</Text>
          <TouchableOpacity onPress={handleUndo} style={styles.undoBtn}>
            <Text style={styles.undoBtnText}>Undo</Text>
          </TouchableOpacity>
        </View>
      )}
    </SafeAreaView>
  );

  const ResultScreen = () => (
    <SafeAreaView style={[styles.safe, { paddingTop: Platform.OS === "android" ? StatusBar.currentHeight || 12 : 0, backgroundColor: (result?.score ?? 0) >= 60 ? "#F1FBF8" : "#FFF7F5" }]}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.resultHeaderRoot}>
        <View style={styles.resultHeaderLeft}>
          <IconButton name="arrow-left" size={22} onPress={() => setView("home")} />
        </View>
        <View style={styles.resultHeaderCenter}>
          <Text style={styles.resultHeaderTitle} numberOfLines={1} adjustsFontSizeToFit>
            Verification Result
          </Text>
        </View>
        <View style={styles.resultHeaderRight} />
      </View>

      <ScrollView contentContainerStyle={styles.resultScrollContent} showsVerticalScrollIndicator={false}>
        {result && <ResultCard result={result} />}

        {imageAsset && (
          <View style={styles.imagePreviewSection}>
            <View style={styles.imagePreviewHeader}>
              <MaterialCommunityIcons name="image" size={18} color={THEME.primary} style={{ marginRight: 8 }} />
              <Text style={styles.imagePreviewTitle}>Scanned Label</Text>
            </View>
            <Image source={{ uri: imageAsset.uri }} style={styles.preview} />
          </View>
        )}

        <View style={styles.actionButtonsContainer}>
          <PrimaryButton label="Scan Another" icon="plus" onPress={() => setView("home")} style={{ flex: 1, marginRight: 8 }} />
          <PrimaryButton label="Share Result" icon="share-variant" onPress={() => Alert.alert("Share", "Coming soon")} secondary style={{ flex: 1, marginLeft: 8 }} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );

  return (
    <View style={{ flex: 1, backgroundColor: THEME.bg }}>
      {loading && (
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color={THEME.primary} />
            <Text style={styles.loadingTitle}>Analyzing Label</Text>
            <Text style={styles.loadingSubtext}>Using AI to verify authenticity...</Text>
          </View>
        </View>
      )}

      {view === "login" && <LoginScreen />}
      {view === "home" && user && <HomeScreen />}
      {view === "result" && result && <ResultScreen />}
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: THEME.bg },

  card: { backgroundColor: THEME.surface, borderRadius: 14, padding: 18, marginHorizontal: 12, shadowColor: THEME.shadow, shadowOpacity: 0.08, shadowRadius: 10, elevation: 6 },

  iconBtn: { padding: 8, borderRadius: 10 },
  primaryBtn: { backgroundColor: THEME.primary, paddingVertical: 12, borderRadius: 12, flexDirection: "row", alignItems: "center", justifyContent: "center" },
  secondaryBtn: { backgroundColor: THEME.surface, borderWidth: 2, borderColor: THEME.primary },
  btnInner: { flexDirection: "row", alignItems: "center", justifyContent: "center" },
  btnText: { color: THEME.surface, fontWeight: "800" },

  header: { height: 180, justifyContent: "center", alignItems: "center", paddingTop: 10 },
  headerOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(7,91,154,0.10)" },
  headerContent: { alignItems: "center", zIndex: 1 },
  logoWrap: { width: 64, height: 64, borderRadius: 16, backgroundColor: THEME.primary, alignItems: "center", justifyContent: "center" },
  appTitle: { color: THEME.surface, fontSize: 22, fontWeight: "900", marginTop: 10 },
  appSubtitle: { color: "rgba(255,255,255,0.95)", marginTop: 6 },

  loginContent: { paddingHorizontal: 8, paddingTop: 16, paddingBottom: 30 },
  cardTitle: { fontSize: 18, fontWeight: "900", color: THEME.text },
  cardSubtitle: { color: THEME.textLight, fontSize: 13, marginBottom: 14 },

  fieldsContainer: { marginTop: 8 },
  field: { flexDirection: "row", alignItems: "center", backgroundColor: THEME.subtle, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, marginBottom: 10, borderWidth: 1, borderColor: THEME.border },
  input: { flex: 1, fontSize: 15, color: THEME.text, marginLeft: 8 },

  disclaimer: { marginTop: 12, color: THEME.muted, fontSize: 12, textAlign: "center" },
  version: { marginTop: 18, textAlign: "center", color: THEME.textLight },

  // NEW: Home header container + medical card
  homeHeaderContainer: { paddingHorizontal: 16, paddingTop: 12, paddingBottom: 8 },
  homeHeaderBg: { width: "100%", borderRadius: 18, overflow: "hidden", padding: 12 },
  homeHeaderOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(7,91,154,0.06)" },
  homeHeaderInner: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  homeHeaderLeft: { flex: 1 },
  homeHeaderRight: { marginLeft: 12 },

  // the medical-blue rounded card that sits behind welcome text
  medicalCard: {
    backgroundColor: THEME.primary,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 14,
    minWidth: 240,
    shadowColor: THEME.shadow,
    shadowOpacity: 0.12,
    shadowRadius: 12,
    elevation: 8,
  },
  welcomeInCard: { color: THEME.surface, fontSize: 13, fontWeight: "700", opacity: 0.95 },
  usernameInCard: { color: THEME.surface, fontSize: 20, fontWeight: "900", marginTop: 6, letterSpacing: 0.2 },
  welcomeSubInCard: { color: "rgba(255,255,255,0.9)", fontSize: 12, marginTop: 6 },

  // home action & history
  actionGrid: { flexDirection: "row", paddingHorizontal: 12, marginBottom: 18 },
  actionCard: { flex: 1, borderRadius: 12, paddingVertical: 14, alignItems: "center", justifyContent: "center", elevation: 3, marginHorizontal: 6, shadowColor: THEME.shadow, shadowOpacity: 0.06, shadowRadius: 8 },
  actionIcon: { width: 64, height: 64, borderRadius: 12, alignItems: "center", justifyContent: "center", marginBottom: 10 },
  actionCardTitle: { fontWeight: "800", color: THEME.text },
  actionCardSub: { fontSize: 11, color: THEME.muted },

  section: { marginHorizontal: 12, marginTop: 14, marginBottom: 20 },
  sectionHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  sectionTitle: { fontWeight: "900", color: THEME.text, fontSize: 16 },
  historyBadge: { backgroundColor: THEME.primary, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 },
  historyBadgeText: { color: THEME.surface, fontWeight: "900", fontSize: 12 },

  historyRow: { backgroundColor: THEME.surface, padding: 12, borderRadius: 10, flexDirection: "row", alignItems: "center", borderLeftWidth: 4, marginBottom: 10, shadowColor: THEME.shadow, shadowOpacity: 0.04, shadowRadius: 6 },
  historyIcon: { width: 44, height: 44, borderRadius: 10, alignItems: "center", justifyContent: "center" },
  historyTitle: { fontWeight: "800", color: THEME.text },
  historyMeta: { color: THEME.muted, marginTop: 4, fontSize: 12 },

  pill: { paddingHorizontal: 10, paddingVertical: 6, borderRadius: 16 },
  pillText: { fontWeight: "900", fontSize: 12 },

  deleteAction: {
    backgroundColor: THEME.danger,
    width: 100,
    marginVertical: 8,
    marginRight: 12,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
  },
  deleteText: { color: "#fff", fontWeight: "900", marginTop: 6 },

  // Result header robust center
  resultHeaderRoot: { height: 56, justifyContent: "center", borderBottomWidth: 0, borderBottomColor: THEME.border },
  resultHeaderLeft: { position: "absolute", left: 12, top: 12, width: 44, height: 44, justifyContent: "center" },
  resultHeaderCenter: { alignItems: "center", justifyContent: "center", marginHorizontal: 60 },
  resultHeaderRight: { position: "absolute", right: 12, top: 12, width: 44, height: 44 },
  resultHeaderTitle: { fontWeight: "900", color: THEME.text, fontSize: 16 },

  resultScrollContent: { paddingHorizontal: 16, paddingBottom: 40 },

  resultCard: { backgroundColor: THEME.surface, borderRadius: 12, overflow: "hidden", elevation: 6, marginTop: 8 },
  resultTop: { padding: 18, alignItems: "center", justifyContent: "center" },
  resultContent: { padding: 14 },

  statusBadge: { flexDirection: "row", alignItems: "center", paddingHorizontal: 8, paddingVertical: 6, borderRadius: 16, alignSelf: "flex-start" },
  statusBadgeText: { fontWeight: "900", fontSize: 13 },

  scoreSection: { flexDirection: "row", alignItems: "center", marginBottom: 12 },
  scoreCircle: { width: 88, height: 88, borderRadius: 44, alignItems: "center", justifyContent: "center", flexDirection: "row" },
  scoreNumber: { fontSize: 28, fontWeight: "900", color: THEME.text },
  scorePercent: { fontSize: 14, color: THEME.muted, marginTop: 10 },

  scoreLabel: { fontSize: 13, color: THEME.muted, fontWeight: "800" },
  scoreSubtext: { fontSize: 12, color: THEME.textLight },

  progressContainer: { marginVertical: 10 },
  progressBg: { height: 8, backgroundColor: THEME.border, borderRadius: 8, overflow: "hidden" },
  progressFill: { height: 8, borderRadius: 8 },

  imagePreviewSection: { marginTop: 18, marginHorizontal: 4 },
  imagePreviewHeader: { flexDirection: "row", alignItems: "center", marginBottom: 10 },
  imagePreviewTitle: { fontWeight: "900", color: THEME.text },
  preview: { width: "100%", height: 220, borderRadius: 12 },

  actionButtonsContainer: { flexDirection: "row", justifyContent: "space-between", paddingHorizontal: 16, marginTop: 20 },

  // Undo bar
  undoBar: {
    position: "absolute",
    left: 12,
    right: 12,
    bottom: 18,
    backgroundColor: "#111",
    opacity: 0.95,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    elevation: 10,
  },
  undoText: { color: "#fff", fontWeight: "700" },
  undoBtn: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8, backgroundColor: "#fff" },
  undoBtnText: { color: "#111", fontWeight: "900" },

  loadingOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(14,38,64,0.24)", justifyContent: "center", alignItems: "center", zIndex: 9999 },
  loadingBox: { backgroundColor: THEME.surface, padding: 16, borderRadius: 12, alignItems: "center" },
  loadingTitle: { marginTop: 12, fontWeight: "800", color: THEME.text },
  loadingSubtext: { marginTop: 6, color: THEME.textLight },
});
