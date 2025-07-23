// IPAI Configuration UI JavaScript

// API Base URL
const API_BASE = '/api/v1';

// LLM Provider Models
const LLM_MODELS = {
    openai: [
        'gpt-4-turbo-preview',
        'gpt-4-1106-preview',
        'gpt-4',
        'gpt-3.5-turbo-1106',
        'gpt-3.5-turbo'
    ],
    anthropic: [
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
        'claude-2.1',
        'claude-2.0',
        'claude-instant-1.2'
    ],
    google: [
        'gemini-pro',
        'gemini-pro-vision',
        'gemini-ultra'
    ],
    cohere: [
        'command-r-plus',
        'command-r',
        'command',
        'command-light'
    ],
    together: [
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'meta-llama/Llama-2-70b-chat-hf',
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
        'zero-one-ai/Yi-34B-Chat'
    ],
    replicate: [
        'meta/llama-2-70b-chat',
        'mistralai/mixtral-8x7b-instruct-v0.1',
        '01-ai/yi-34b-chat'
    ],
    huggingface: [
        'meta-llama/Llama-2-70b-chat-hf',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'tiiuae/falcon-180B-chat'
    ],
    groq: [
        'llama2-70b-4096',
        'mixtral-8x7b-32768',
        'gemma-7b-it'
    ],
    mistral: [
        'mistral-large-latest',
        'mistral-medium-latest',
        'mistral-small-latest',
        'mistral-embed'
    ],
    ollama: [
        'llama2',
        'mistral',
        'codellama',
        'neural-chat',
        'starling-lm',
        'orca-mini'
    ],
    custom: []
};

// State
let currentLLMProviders = [];
let currentWallets = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAuthentication();
    loadCurrentConfig();
    setupEventListeners();
    updateLLMFields();
});

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('[id$="-tab"]').forEach(tab => {
        tab.classList.remove('tab-active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-content`).style.display = 'block';
    document.getElementById(`${tabName}-tab`).classList.add('tab-active');
}

// Authentication Check
async function checkAuthentication() {
    try {
        const response = await fetch(`${API_BASE}/auth/me`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (!response.ok) {
            window.location.href = '/login.html';
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        window.location.href = '/login.html';
    }
}

// Load Current Configuration
async function loadCurrentConfig() {
    try {
        // Load LLM providers
        const llmResponse = await fetch(`${API_BASE}/config/llm-providers`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (llmResponse.ok) {
            currentLLMProviders = await llmResponse.json();
            displayLLMProviders();
            updateDefaultLLMOptions();
        }
        
        // Load wallets
        const walletResponse = await fetch(`${API_BASE}/config/wallets`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (walletResponse.ok) {
            currentWallets = await walletResponse.json();
            displayWallets();
            updateDefaultWalletOptions();
            updateSAGEBalance();
        }
        
        // Load settings
        const settingsResponse = await fetch(`${API_BASE}/config/settings`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (settingsResponse.ok) {
            const settings = await settingsResponse.json();
            document.getElementById('auto-connect-wallet').checked = settings.auto_connect_wallet;
            document.getElementById('coherence-tracking').checked = settings.coherence_tracking;
            document.getElementById('default-llm').value = settings.default_llm || '';
            document.getElementById('default-wallet').value = settings.default_wallet || '';
        }
        
    } catch (error) {
        console.error('Failed to load configuration:', error);
        showToast('Failed to load configuration', 'error');
    }
}

// LLM Provider Management
function updateLLMFields() {
    const provider = document.getElementById('llm-provider').value;
    const modelSelect = document.getElementById('llm-model');
    const apiKeyField = document.getElementById('api-key-field');
    const apiBaseField = document.getElementById('api-base-field');
    
    // Clear models
    modelSelect.innerHTML = '';
    
    // Add models for selected provider
    const models = LLM_MODELS[provider] || [];
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    });
    
    // Show/hide fields based on provider
    if (provider === 'ollama') {
        apiKeyField.style.display = 'none';
        apiBaseField.style.display = 'block';
        document.getElementById('llm-api-base').value = 'http://localhost:11434';
    } else if (provider === 'custom') {
        apiKeyField.style.display = 'block';
        apiBaseField.style.display = 'block';
    } else {
        apiKeyField.style.display = 'block';
        apiBaseField.style.display = 'none';
    }
}

function displayLLMProviders() {
    const container = document.getElementById('llm-list');
    container.innerHTML = '';
    
    if (currentLLMProviders.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-4">No LLM providers configured</p>';
        return;
    }
    
    currentLLMProviders.forEach(provider => {
        const card = document.createElement('div');
        card.className = 'provider-card bg-gray-50 p-4 rounded-lg hover:bg-gray-100';
        card.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <h3 class="font-semibold text-gray-800">${provider.name}</h3>
                    <p class="text-sm text-gray-600">${provider.provider_name}</p>
                    <p class="text-xs text-gray-500">${provider.model}</p>
                </div>
                <div class="flex items-center space-x-2">
                    ${provider.active ? '<span class="pulse-dot w-2 h-2 bg-green-500 rounded-full"></span>' : ''}
                    <button onclick="testLLMProvider('${provider.name}')" class="text-blue-600 hover:text-blue-800">
                        <i class="fas fa-vial"></i>
                    </button>
                    <button onclick="removeLLMProvider('${provider.name}')" class="text-red-600 hover:text-red-800">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

async function testLLMProvider(name) {
    try {
        const response = await fetch(`${API_BASE}/config/llm-providers/${name}/test`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (response.ok) {
            showToast('LLM provider test successful!', 'success');
        } else {
            showToast('LLM provider test failed', 'error');
        }
    } catch (error) {
        showToast('Failed to test LLM provider', 'error');
    }
}

async function removeLLMProvider(name) {
    if (!confirm(`Remove LLM provider "${name}"?`)) return;
    
    try {
        const response = await fetch(`${API_BASE}/config/llm-providers/${name}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (response.ok) {
            showToast('LLM provider removed', 'success');
            loadCurrentConfig();
        } else {
            showToast('Failed to remove LLM provider', 'error');
        }
    } catch (error) {
        showToast('Failed to remove LLM provider', 'error');
    }
}

// Wallet Management
function updateWalletFields() {
    const walletType = document.getElementById('wallet-type').value;
    const privateKeyField = document.getElementById('private-key-field');
    const connectButton = document.getElementById('connect-wallet-button');
    
    if (walletType === 'private_key') {
        privateKeyField.style.display = 'block';
        connectButton.classList.add('hidden');
    } else if (['metamask', 'walletconnect', 'coinbase', 'phantom', 'rainbow', 'trust'].includes(walletType)) {
        privateKeyField.style.display = 'none';
        connectButton.classList.remove('hidden');
    } else {
        privateKeyField.style.display = 'none';
        connectButton.classList.add('hidden');
    }
}

function displayWallets() {
    const container = document.getElementById('wallet-list');
    container.innerHTML = '';
    
    if (currentWallets.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-4">No wallets connected</p>';
        return;
    }
    
    currentWallets.forEach(wallet => {
        const card = document.createElement('div');
        card.className = 'provider-card bg-gray-50 p-4 rounded-lg hover:bg-gray-100';
        card.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <h3 class="font-semibold text-gray-800">${wallet.name}</h3>
                    <p class="text-sm text-gray-600">${wallet.type} - ${wallet.network}</p>
                    <p class="text-xs text-gray-500 font-mono">${shortenAddress(wallet.address)}</p>
                </div>
                <div class="flex items-center space-x-2">
                    ${wallet.active ? '<span class="pulse-dot w-2 h-2 bg-green-500 rounded-full"></span>' : ''}
                    <button onclick="removeWallet('${wallet.name}')" class="text-red-600 hover:text-red-800">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

async function connectBrowserWallet() {
    const walletType = document.getElementById('wallet-type').value;
    
    // This would integrate with web3modal or similar
    showToast('Browser wallet connection coming soon', 'info');
}

async function removeWallet(name) {
    if (!confirm(`Remove wallet "${name}"?`)) return;
    
    try {
        const response = await fetch(`${API_BASE}/config/wallets/${name}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (response.ok) {
            showToast('Wallet removed', 'success');
            loadCurrentConfig();
        } else {
            showToast('Failed to remove wallet', 'error');
        }
    } catch (error) {
        showToast('Failed to remove wallet', 'error');
    }
}

async function updateSAGEBalance() {
    try {
        const response = await fetch(`${API_BASE}/blockchain/sage-balance`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            document.getElementById('sage-balance').textContent = `${data.balance.toFixed(2)} SAGE`;
        }
    } catch (error) {
        console.error('Failed to fetch SAGE balance:', error);
    }
}

async function claimDailySAGE() {
    try {
        const response = await fetch(`${API_BASE}/blockchain/claim-sage`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            }
        });
        
        if (response.ok) {
            showToast('Daily SAGE reward claimed!', 'success');
            updateSAGEBalance();
        } else {
            const error = await response.json();
            showToast(error.detail || 'Failed to claim reward', 'error');
        }
    } catch (error) {
        showToast('Failed to claim reward', 'error');
    }
}

// Form Handlers
function setupEventListeners() {
    // LLM form
    document.getElementById('llm-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = {
            name: document.getElementById('llm-name').value,
            provider: document.getElementById('llm-provider').value,
            api_key: document.getElementById('llm-api-key').value,
            api_base: document.getElementById('llm-api-base').value || null,
            model: document.getElementById('llm-model').value
        };
        
        try {
            const response = await fetch(`${API_BASE}/config/llm-providers`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
                },
                body: JSON.stringify(formData)
            });
            
            if (response.ok) {
                showToast('LLM provider added successfully', 'success');
                document.getElementById('llm-form').reset();
                loadCurrentConfig();
            } else {
                const error = await response.json();
                showToast(error.detail || 'Failed to add LLM provider', 'error');
            }
        } catch (error) {
            showToast('Failed to add LLM provider', 'error');
        }
    });
    
    // Wallet form
    document.getElementById('wallet-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = {
            name: document.getElementById('wallet-name').value,
            wallet_type: document.getElementById('wallet-type').value,
            network: document.getElementById('wallet-network').value,
            private_key: document.getElementById('wallet-private-key').value || null
        };
        
        try {
            const response = await fetch(`${API_BASE}/config/wallets`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
                },
                body: JSON.stringify(formData)
            });
            
            if (response.ok) {
                showToast('Wallet added successfully', 'success');
                document.getElementById('wallet-form').reset();
                loadCurrentConfig();
            } else {
                const error = await response.json();
                showToast(error.detail || 'Failed to add wallet', 'error');
            }
        } catch (error) {
            showToast('Failed to add wallet', 'error');
        }
    });
}

// Settings
async function saveSettings() {
    const settings = {
        auto_connect_wallet: document.getElementById('auto-connect-wallet').checked,
        coherence_tracking: document.getElementById('coherence-tracking').checked,
        default_llm: document.getElementById('default-llm').value || null,
        default_wallet: document.getElementById('default-wallet').value || null
    };
    
    try {
        const response = await fetch(`${API_BASE}/config/settings`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            },
            body: JSON.stringify(settings)
        });
        
        if (response.ok) {
            showToast('Settings saved successfully', 'success');
        } else {
            showToast('Failed to save settings', 'error');
        }
    } catch (error) {
        showToast('Failed to save settings', 'error');
    }
}

async function changePassword() {
    const oldPassword = prompt('Enter current master password:');
    if (!oldPassword) return;
    
    const newPassword = prompt('Enter new master password:');
    if (!newPassword) return;
    
    const confirmPassword = prompt('Confirm new master password:');
    if (newPassword !== confirmPassword) {
        showToast('Passwords do not match', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/config/change-password`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('ipai_token')}`
            },
            body: JSON.stringify({
                old_password: oldPassword,
                new_password: newPassword
            })
        });
        
        if (response.ok) {
            showToast('Password changed successfully', 'success');
        } else {
            showToast('Failed to change password', 'error');
        }
    } catch (error) {
        showToast('Failed to change password', 'error');
    }
}

// Utility Functions
function updateDefaultLLMOptions() {
    const select = document.getElementById('default-llm');
    select.innerHTML = '<option value="">None</option>';
    
    currentLLMProviders.forEach(provider => {
        const option = document.createElement('option');
        option.value = provider.name;
        option.textContent = provider.name;
        select.appendChild(option);
    });
}

function updateDefaultWalletOptions() {
    const select = document.getElementById('default-wallet');
    select.innerHTML = '<option value="">None</option>';
    
    currentWallets.forEach(wallet => {
        const option = document.createElement('option');
        option.value = wallet.name;
        option.textContent = wallet.name;
        select.appendChild(option);
    });
}

function togglePasswordVisibility(fieldId) {
    const field = document.getElementById(fieldId);
    const icon = field.nextElementSibling.querySelector('i');
    
    if (field.type === 'password') {
        field.type = 'text';
        icon.classList.replace('fa-eye', 'fa-eye-slash');
    } else {
        field.type = 'password';
        icon.classList.replace('fa-eye-slash', 'fa-eye');
    }
}

function shortenAddress(address) {
    if (!address) return '';
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    
    const bgColor = {
        'success': 'bg-green-500',
        'error': 'bg-red-500',
        'warning': 'bg-yellow-500',
        'info': 'bg-blue-500'
    }[type] || 'bg-gray-500';
    
    toast.className = `${bgColor} text-white px-6 py-3 rounded-lg shadow-lg mb-2 transform transition-all duration-300 translate-x-full`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.classList.remove('translate-x-full');
    }, 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.add('translate-x-full');
        setTimeout(() => {
            container.removeChild(toast);
        }, 300);
    }, 3000);
}