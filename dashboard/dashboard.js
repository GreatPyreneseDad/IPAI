/**
 * IPAI Dashboard Main JavaScript
 * 
 * Handles real-time updates, chat interface, metrics display,
 * and integration with the helical blockchain system.
 */

class IPAIDashboard {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.socket = null;
        this.isConnected = false;
        this.autoRotate = false;
        this.metrics = {
            psi: [],
            rho: [],
            q: [],
            f: []
        };
        this.charts = {};
        
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing IPAI Dashboard...');
        
        // Initialize event listeners
        this.setupEventListeners();
        
        // Initialize charts
        this.initializeCharts();
        
        // Check system status
        await this.checkSystemStatus();
        
        // Start real-time updates
        this.startRealTimeUpdates();
        
        // Load initial data
        await this.loadInitialData();
        
        console.log('✅ IPAI Dashboard initialized successfully');
    }
    
    setupEventListeners() {
        // Chat functionality
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const clearBtn = document.getElementById('clear-chat');
        
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
        
        sendBtn.addEventListener('click', () => this.sendMessage());
        clearBtn.addEventListener('click', () => this.clearChat());
        
        // Visualization controls
        document.getElementById('rotate-helix').addEventListener('click', () => {
            this.toggleAutoRotate();
        });
        
        document.getElementById('reset-view').addEventListener('click', () => {
            if (window.helixVisualization) {
                window.helixVisualization.resetView();
            }
        });
        
        // Strand filter
        document.getElementById('strand-filter').addEventListener('change', (e) => {
            this.filterActivity(e.target.value);
        });
        
        // Modal controls
        document.getElementById('close-modal').addEventListener('click', () => {
            document.getElementById('block-modal').style.display = 'none';
        });
        
        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('block-modal');
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
        
        // Web3 wallet connection
        document.getElementById('connect-wallet').addEventListener('click', () => {
            if (window.web3Integration) {
                window.web3Integration.connectWallet();
            }
        });
    }
    
    initializeCharts() {
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        display: false,
                        min: 0,
                        max: 1
                    }
                },
                elements: {
                    point: {
                        radius: 0
                    },
                    line: {
                        tension: 0.4,
                        borderWidth: 2
                    }
                },
                animation: {
                    duration: 500
                }
            }
        };
        
        // Initialize metric charts
        const metrics = ['psi', 'rho', 'q', 'f'];
        const colors = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b'];
        
        metrics.forEach((metric, index) => {
            const ctx = document.getElementById(`${metric}-chart`).getContext('2d');
            this.charts[metric] = new Chart(ctx, {
                ...chartConfig,
                data: {
                    labels: Array(20).fill(''),
                    datasets: [{
                        data: Array(20).fill(0.5),
                        borderColor: colors[index],
                        backgroundColor: colors[index] + '20',
                        fill: true
                    }]
                }
            });
        });
    }
    
    async checkSystemStatus() {
        try {
            // Check IPAI server
            const response = await fetch(`${this.apiBase}/health`);
            const serverStatus = document.getElementById('server-status');
            
            if (response.ok) {
                serverStatus.className = 'status-indicator online';
                this.isConnected = true;
            } else {
                serverStatus.className = 'status-indicator offline';
            }
            
            // Check LLM status
            const statusResponse = await fetch(`${this.apiBase}/api/v1/status`);
            if (statusResponse.ok) {
                const statusData = await statusResponse.json();
                const llmStatus = document.getElementById('llm-status');
                const blockchainStatus = document.getElementById('blockchain-status');
                
                llmStatus.className = statusData.ollama_connected ? 
                    'status-indicator online' : 'status-indicator offline';
                    
                blockchainStatus.className = statusData.system === 'IPAI' ? 
                    'status-indicator online' : 'status-indicator warning';
            }
        } catch (error) {
            console.error('Failed to check system status:', error);
            document.getElementById('server-status').className = 'status-indicator offline';
            document.getElementById('llm-status').className = 'status-indicator offline';
            document.getElementById('blockchain-status').className = 'status-indicator offline';
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        this.addMessageToChat('user', message);
        input.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Send to IPAI backend
            const response = await fetch(`${this.apiBase}/api/v1/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    context: {
                        dashboard: true,
                        timestamp: new Date().toISOString()
                    }
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                
                // Remove typing indicator
                this.hideTypingIndicator();
                
                // Add assistant response
                this.addMessageToChat('assistant', data.response);
                
                // Log interaction to blockchain
                await this.logInteractionToBlockchain('chat', {
                    user_message: message,
                    assistant_response: data.response,
                    timestamp: data.timestamp
                });
                
                // Update metrics if available
                if (data.coherence_analysis) {
                    this.updateMetrics(data.coherence_analysis);
                }
            } else {
                this.hideTypingIndicator();
                this.addMessageToChat('assistant', 'Sorry, I encountered an error. Please try again.');
            }
        } catch (error) {
            console.error('Failed to send message:', error);
            this.hideTypingIndicator();
            this.addMessageToChat('assistant', 'Connection error. Please check your network and try again.');
        }
    }
    
    addMessageToChat(sender, message) {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;
        
        const timestamp = new Date().toLocaleTimeString();
        messageDiv.innerHTML = `
            <div class="message-content">${message}</div>
            <div class="message-time">${timestamp}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    showTypingIndicator() {
        const chatMessages = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message assistant typing';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        `;
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    clearChat() {
        document.getElementById('chat-messages').innerHTML = '';
        this.addMessageToChat('assistant', 'Hello! I\\'m your GCT-enhanced AI assistant. How can I help you build coherence today?');
    }
    
    async logInteractionToBlockchain(type, data) {
        try {
            const response = await fetch(`${this.apiBase}/api/v1/blockchain/log`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    interaction_type: type,
                    data: data,
                    user_id: 'dashboard_user',
                    timestamp: new Date().toISOString()
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Interaction logged to blockchain:', result.transaction_id);
                
                // Update activity log
                this.addActivityItem(type, data, result.timestamp);
                
                // Refresh blockchain visualization
                if (window.helixVisualization) {
                    window.helixVisualization.refresh();
                }
            }
        } catch (error) {
            console.error('Failed to log interaction to blockchain:', error);
        }
    }
    
    addActivityItem(type, data, timestamp) {
        const activityLog = document.getElementById('activity-log');
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        
        const time = new Date(timestamp).toLocaleTimeString();
        const typeDisplay = type.charAt(0).toUpperCase() + type.slice(1);
        
        let details = '';
        if (type === 'chat') {
            details = `User asked: "${data.user_message.substring(0, 50)}..."`;
        } else {
            details = JSON.stringify(data).substring(0, 100) + '...';
        }
        
        activityItem.innerHTML = `
            <div class="activity-header">
                <span class="activity-type">
                    <span class="strand-indicator interaction"></span>
                    ${typeDisplay}
                </span>
                <span class="activity-time">${time}</span>
            </div>
            <div class="activity-details">${details}</div>
        `;
        
        // Add to top of log
        activityLog.insertBefore(activityItem, activityLog.firstChild);
        
        // Limit to 50 items
        while (activityLog.children.length > 50) {
            activityLog.removeChild(activityLog.lastChild);
        }
    }
    
    updateMetrics(coherenceData) {
        // Update metric values
        document.getElementById('psi-value').textContent = coherenceData.psi_score.toFixed(2);
        document.getElementById('rho-value').textContent = coherenceData.rho_score.toFixed(2);
        document.getElementById('q-value').textContent = coherenceData.q_score.toFixed(2);
        document.getElementById('f-value').textContent = coherenceData.f_score.toFixed(2);
        
        // Calculate and update Soul Echo
        const soulEcho = coherenceData.psi_score * coherenceData.rho_score * 
                        coherenceData.q_score * coherenceData.f_score;
        document.getElementById('soul-echo').textContent = soulEcho.toFixed(3);
        document.getElementById('soul-echo-fill').style.width = `${soulEcho * 100}%`;
        
        // Update charts
        this.updateMetricCharts(coherenceData);
    }
    
    updateMetricCharts(coherenceData) {
        const metrics = ['psi', 'rho', 'q', 'f'];
        
        metrics.forEach(metric => {
            const value = coherenceData[`${metric}_score`];
            const chart = this.charts[metric];
            
            if (chart) {
                // Add new data point
                chart.data.datasets[0].data.push(value);
                chart.data.datasets[0].data.shift(); // Remove oldest point
                chart.update('none'); // Update without animation
            }
        });
    }
    
    filterActivity(strandType) {
        const activityItems = document.querySelectorAll('.activity-item');
        
        activityItems.forEach(item => {
            if (strandType === 'all') {
                item.style.display = 'block';
            } else {
                const indicator = item.querySelector('.strand-indicator');
                if (indicator && indicator.classList.contains(strandType)) {
                    item.style.display = 'block';
                } else {
                    item.style.display = 'none';
                }
            }
        });
    }
    
    toggleAutoRotate() {
        this.autoRotate = !this.autoRotate;
        const button = document.getElementById('rotate-helix');
        
        if (this.autoRotate) {
            button.textContent = 'Stop Rotation';
            button.classList.add('active');
            if (window.helixVisualization) {
                window.helixVisualization.startAutoRotation();
            }
        } else {
            button.textContent = 'Auto Rotate';
            button.classList.remove('active');
            if (window.helixVisualization) {
                window.helixVisualization.stopAutoRotation();
            }
        }
    }
    
    startRealTimeUpdates() {
        // Update system status every 30 seconds
        setInterval(() => {
            this.checkSystemStatus();
        }, 30000);
        
        // Update blockchain data every 10 seconds
        setInterval(() => {
            this.refreshBlockchainData();
        }, 10000);
        
        // Update metrics every 5 seconds
        setInterval(() => {
            this.refreshMetrics();
        }, 5000);
    }
    
    async refreshBlockchainData() {
        try {
            const response = await fetch(`${this.apiBase}/api/v1/blockchain/state`);
            if (response.ok) {
                const data = await response.json();
                this.updateBlockchainStats(data);
            }
        } catch (error) {
            console.error('Failed to refresh blockchain data:', error);
        }
    }
    
    updateBlockchainStats(data) {
        // Update blockchain info display
        document.getElementById('total-blocks').textContent = data.network_info.total_blocks;
        document.getElementById('active-strands').textContent = data.network_info.total_strands;
        document.getElementById('cross-links').textContent = data.network_info.cross_links;
        
        // Update strand cards
        this.updateStrandCards(data.strands);
    }
    
    updateStrandCards(strands) {
        const strandsGrid = document.getElementById('strands-grid');
        strandsGrid.innerHTML = '';
        
        Object.entries(strands).forEach(([strandName, strandData]) => {
            const card = document.createElement('div');
            card.className = 'strand-card';
            
            const latestBlock = strandData.latest_block;
            const coherenceScore = latestBlock ? latestBlock.coherence_score : 0;
            
            card.innerHTML = `
                <div class="strand-header">
                    <span class="strand-name">
                        <span class="strand-indicator ${strandName}"></span>
                        ${strandName.charAt(0).toUpperCase() + strandName.slice(1)}
                    </span>
                    <span class="strand-count">${strandData.block_count} blocks</span>
                </div>
                <div class="strand-stats">
                    <span>Coherence: ${coherenceScore.toFixed(2)}</span>
                    <span>Latest: ${latestBlock ? latestBlock.index : 0}</span>
                </div>
            `;
            
            strandsGrid.appendChild(card);
        });
    }
    
    async refreshMetrics() {
        // Simulate metric updates for demo
        // In a real system, this would fetch from the backend
        const simulatedMetrics = {
            psi_score: 0.7 + Math.random() * 0.3,
            rho_score: 0.6 + Math.random() * 0.4,
            q_score: 0.8 + Math.random() * 0.2,
            f_score: 0.7 + Math.random() * 0.3
        };
        
        this.updateMetrics(simulatedMetrics);
    }
    
    async loadInitialData() {
        // Load initial chat message
        this.addMessageToChat('assistant', 'Welcome to IPAI! I\\'m your GCT-enhanced AI assistant. I help you build coherence through internal consistency (Ψ), accumulated wisdom (ρ), moral activation (q), and social belonging (f). How can I assist you today?');
        
        // Load blockchain data
        await this.refreshBlockchainData();
        
        // Initialize visualization
        if (window.helixVisualization) {
            window.helixVisualization.init();
        }
        
        // Initialize Web3 if available
        if (window.web3Integration) {
            window.web3Integration.init();
        }
    }
    
    showBlockDetails(blockHash) {
        // Show modal with block details
        const modal = document.getElementById('block-modal');
        const details = document.getElementById('block-details');
        
        // Fetch block details and display
        // This would be implemented with actual blockchain data
        details.innerHTML = `
            <h3>Block Details</h3>
            <p><strong>Hash:</strong> ${blockHash}</p>
            <p><strong>Status:</strong> Loading...</p>
        `;
        
        modal.style.display = 'block';
    }
}

// CSS for typing indicator
const typingCSS = `
.typing-dots {
    display: flex;
    gap: 4px;
    padding: 8px;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--text-secondary);
    animation: typing 1.4s infinite;
}

.typing-dots span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dots span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 80%, 100% {
        opacity: 0.3;
    }
    40% {
        opacity: 1;
    }
}
`;

// Add typing CSS to document
const style = document.createElement('style');
style.textContent = typingCSS;
document.head.appendChild(style);

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new IPAIDashboard();
});