// Theme Management Utility

const THEME_KEY = 'neurocanvas-theme';

const getInitialTheme = () => {
  const savedTheme = localStorage.getItem(THEME_KEY);
  if (savedTheme) {
    return savedTheme;
  }

  // Check system preference
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark';
  }

  return 'dark'; // Default to dark as per current design
};

const setTheme = (theme) => {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_KEY, theme);
};

const toggleTheme = () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  setTheme(newTheme);
  updateThemeIcon(newTheme);
};

const updateThemeIcon = (theme) => {
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    themeToggle.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
  }
};

// Initialize theme on page load
const initTheme = () => {
  const theme = getInitialTheme();
  setTheme(theme);
  updateThemeIcon(theme);

  // Add toggle button to header if it doesn't exist
  const header = document.querySelector('.site-header');
  const navLinks = document.querySelector('.nav-links');

  if (header && navLinks && !document.getElementById('theme-toggle')) {
    const themeButton = document.createElement('button');
    themeButton.id = 'theme-toggle';
    themeButton.className = 'theme-toggle';
    themeButton.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
    themeButton.textContent = theme === 'dark' ? '☀️' : '🌙';
    themeButton.onclick = toggleTheme;

    navLinks.appendChild(themeButton);
  }
};

// Listen for system theme changes
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    const savedTheme = localStorage.getItem(THEME_KEY);
    // Only update if user hasn't manually set a preference
    if (!savedTheme) {
      const newTheme = e.matches ? 'dark' : 'light';
      setTheme(newTheme);
      updateThemeIcon(newTheme);
    }
  });
}

// Initialize theme immediately
initTheme();

// Export for use in other scripts if needed
if (typeof window !== 'undefined') {
  window.themeUtils = {
    toggleTheme,
    setTheme,
    getInitialTheme
  };
}

