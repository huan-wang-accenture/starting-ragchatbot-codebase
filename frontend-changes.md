# Frontend Changes: Dark/Light Theme Toggle

## Overview
Added a theme toggle button that allows users to switch between dark and light themes with smooth transitions and persistent preference storage.

## Files Modified

### 1. `frontend/style.css`

**Changes:**
- Added light theme CSS variables under `[data-theme="light"]` selector
- Added new CSS variables for better theme support: `--code-bg`, `--source-link-color`, `--source-link-hover`
- Updated code block styles to use `var(--code-bg)` instead of hardcoded colors
- Updated source link colors to use CSS variables
- Added theme toggle button styles (`.theme-toggle` class)
- Added CSS transitions for smooth theme switching

**New CSS Variables (Light Theme):**
```css
[data-theme="light"] {
    --background: #f8fafc;
    --surface: #ffffff;
    --surface-hover: #f1f5f9;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --code-bg: rgba(0, 0, 0, 0.05);
    --source-link-color: #2563eb;
    --source-link-hover: #1d4ed8;
}
```

### 2. `frontend/index.html`

**Changes:**
- Added theme toggle button with sun/moon SVG icons
- Button is positioned fixed in top-right corner
- Includes proper accessibility attributes (`aria-label`, `title`)
- Updated CSS/JS version numbers for cache invalidation

**New HTML:**
```html
<button id="themeToggle" class="theme-toggle" aria-label="Toggle theme">
    <svg class="sun-icon">...</svg>
    <svg class="moon-icon">...</svg>
</button>
```

### 3. `frontend/script.js`

**Changes:**
- Added `themeToggle` DOM element reference
- Added event listener for theme toggle button
- Added three new functions:
  - `initializeTheme()` - Loads saved theme preference on page load
  - `toggleTheme()` - Switches between dark and light themes
  - `setTheme(theme)` - Applies theme and saves to localStorage

## Features

1. **Toggle Button Design**
   - Circular button positioned in top-right corner
   - Sun icon displayed in dark mode (click to switch to light)
   - Moon icon displayed in light mode (click to switch to dark)
   - Hover and focus states with smooth animations
   - Keyboard accessible (focusable and activatable)

2. **Theme Persistence**
   - Theme preference saved to localStorage
   - Preference restored on page reload
   - Defaults to dark theme if no preference saved

3. **Smooth Transitions**
   - 0.3s ease transition on background colors, text colors, and borders
   - Button icon rotation on hover
   - Scale animation on button interaction

4. **Accessibility**
   - Proper `aria-label` for screen readers
   - `title` attribute for tooltip
   - Focus ring for keyboard navigation
   - Good color contrast in both themes
