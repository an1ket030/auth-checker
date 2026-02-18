---
name: expo-build
description: Build and deploy the AuthChecker Expo/React Native app using EAS Build, including APK/AAB generation and common troubleshooting.
---

# Expo Build Skill

## Project Context

- **App Name**: AuthChecker
- **Slug**: `auth-checker`
- **Package**: `com.student.authchecker.demo`
- **Primary App Dir**: `mobile/` (preferred — has `expo-build-properties` plugin)
- **Legacy App Dir**: `app/` (older config, avoid unless specifically asked)
- **EAS Config**: `app/eas.json`
- **Min SDK**: 24 (Android)
- **Cleartext Traffic**: Enabled (required for local network dev)

## Prerequisites

Before building, ensure:

1. **EAS CLI is installed globally**:
   ```bash
   npm install -g eas-cli
   ```

2. **User is logged in to EAS**:
   ```bash
   eas login
   ```

3. **Dependencies are installed** in the app directory:
   ```bash
   cd mobile
   npm install
   ```

4. **EAS project is linked** (project ID in `app.json > extra.eas.projectId`):
   - `mobile/`: `ab94bfb7-f08d-4085-b2a4-199aefb2c356`
   - `app/`: `171e8b31-db16-4be9-bdc4-24ba5c64a56a`

## Build Commands

### Preview Build (APK — for testing/sharing)
```bash
cd mobile
eas build --platform android --profile preview
```
This produces a `.apk` file that can be directly installed on Android devices.

### Production Build (AAB — for Play Store)
```bash
cd mobile
eas build --platform android --profile production
```
This produces an `.aab` file for Google Play Store submission.

### Local Build (no EAS servers — requires Android SDK)
```bash
cd mobile
eas build --platform android --profile preview --local
```
Builds locally on the machine. Requires Java 17+ and Android SDK installed.

### Check Build Status
```bash
eas build:list
```

## EAS Configuration Reference

The `eas.json` should be in the app directory being built from. Current config:
```json
{
  "cli": { "version": ">= 3.0.0" },
  "build": {
    "preview": {
      "android": { "buildType": "apk" }
    },
    "production": {
      "android": { "buildType": "app-bundle" }
    }
  }
}
```

## Common Issues & Fixes

### 1. Network Errors / Cleartext Traffic Blocked
**Symptom**: App can't connect to backend over HTTP on local network.
**Fix**: Ensure both of these are set in `app.json`:
```json
"android": {
  "usesCleartextTraffic": true
}
```
AND the `expo-build-properties` plugin is configured:
```json
"plugins": [
  ["expo-build-properties", {
    "android": {
      "usesCleartextTraffic": true
    }
  }]
]
```

### 2. Build Fails — Missing `eas.json`
**Symptom**: `eas.json not found`
**Fix**: Ensure `eas.json` exists in the directory you're running from. Copy from `app/eas.json` if needed:
```bash
cp ../app/eas.json ./eas.json
```

### 3. SDK Version Mismatch
**Symptom**: Build fails with SDK version errors.
**Fix**: Check `sdkVersion` in `app.json` matches installed `expo` package version:
```bash
npx expo --version
```

### 4. Prebuild / Native Code Issues
If you need to eject or inspect native code:
```bash
cd mobile
npx expo prebuild --platform android
```

### 5. Cache Issues
Clear all Expo and EAS caches:
```bash
npx expo start --clear
eas build --platform android --profile preview --clear-cache
```

### 6. Credentials / Keystore
- EAS manages Android keystores automatically for preview builds
- For production, you may need to provide your own keystore:
  ```bash
  eas credentials
  ```

## Environment Variables

If the app needs environment variables at build time, configure them in `eas.json`:
```json
{
  "build": {
    "preview": {
      "env": {
        "API_URL": "http://YOUR_IP:8000"
      }
    }
  }
}
```

## Post-Build Steps

1. **Download APK**: After build completes, EAS provides a download URL
2. **Install on device**: Transfer APK to phone or use `adb install <path-to-apk>`
3. **Test connectivity**: Ensure phone and backend server are on the same network
4. **Backend must be running**: Start with `python -m backend.main` or `run_server.bat`
