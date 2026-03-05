/**
 * Web3 Integration for IPAI Dashboard
 * 
 * Handles wallet connections, blockchain interactions,
 * and integration with the helical blockchain system.
 */

class Web3Integration {
    constructor() {
        this.web3 = null;
        this.account = null;
        this.networkId = null;
        this.isConnected = false;
        
        // Contract addresses (would be deployed contracts in production)
        this.contracts = {
            coherenceNFT: null,
            wisdomToken: null,
            ipaiDAO: null
        };
        
        this.init();
    }
    
    init() {
        console.log('🌐 Initializing Web3 integration...');
        
        // Check if MetaMask is installed
        if (typeof window.ethereum !== 'undefined') {
            this.web3 = new Web3(window.ethereum);
            this.setupEventListeners();
            this.checkConnection();
        } else {
            console.log('MetaMask not detected');
            this.showInstallPrompt();
        }
    }
    
    setupEventListeners() {
        // Account change
        window.ethereum.on('accountsChanged', (accounts) => {
            if (accounts.length === 0) {
                this.disconnect();
            } else {
                this.account = accounts[0];
                this.updateUI();
            }
        });
        
        // Network change
        window.ethereum.on('chainChanged', (chainId) => {
            this.networkId = parseInt(chainId, 16);
            this.updateUI();
            window.location.reload(); // Recommended by MetaMask
        });
        
        // Connection
        window.ethereum.on('connect', (connectInfo) => {
            console.log('Connected to network:', connectInfo.chainId);
        });
        
        // Disconnection
        window.ethereum.on('disconnect', (error) => {
            console.log('Disconnected from network:', error);
            this.disconnect();
        });
    }
    
    async checkConnection() {
        try {
            const accounts = await this.web3.eth.getAccounts();
            if (accounts.length > 0) {
                this.account = accounts[0];
                this.networkId = await this.web3.eth.net.getId();
                this.isConnected = true;
                this.updateUI();
            }
        } catch (error) {
            console.error('Failed to check connection:', error);
        }
    }
    
    async connectWallet() {
        try {
            // Request account access
            const accounts = await window.ethereum.request({
                method: 'eth_requestAccounts'
            });
            
            if (accounts.length > 0) {
                this.account = accounts[0];
                this.networkId = await this.web3.eth.net.getId();
                this.isConnected = true;
                
                // Get balance
                const balance = await this.web3.eth.getBalance(this.account);
                this.balance = this.web3.utils.fromWei(balance, 'ether');
                
                this.updateUI();
                this.enableWeb3Features();
                
                console.log('✅ Wallet connected:', this.account);
                
                // Log wallet connection to blockchain
                await this.logToHelicalBlockchain('wallet_connection', {
                    address: this.account,
                    network_id: this.networkId,
                    balance: this.balance
                });
                
            }
        } catch (error) {
            console.error('Failed to connect wallet:', error);
            this.showError('Failed to connect wallet. Please try again.');
        }
    }
    
    disconnect() {
        this.account = null;
        this.networkId = null;
        this.isConnected = false;
        this.balance = null;
        this.updateUI();
        this.disableWeb3Features();
    }
    
    updateUI() {
        const connectButton = document.getElementById('connect-wallet');
        const walletInfo = document.getElementById('wallet-info');
        const walletAddress = document.getElementById('wallet-address');
        const networkName = document.getElementById('network-name');
        const walletBalance = document.getElementById('wallet-balance');
        
        if (this.isConnected && this.account) {
            connectButton.textContent = 'Disconnect';
            connectButton.onclick = () => this.disconnect();
            
            walletInfo.style.display = 'block';
            walletAddress.textContent = this.formatAddress(this.account);
            networkName.textContent = this.getNetworkName(this.networkId);
            walletBalance.textContent = `${parseFloat(this.balance).toFixed(4)} ETH`;
            
        } else {
            connectButton.textContent = 'Connect Wallet';
            connectButton.onclick = () => this.connectWallet();
            
            walletInfo.style.display = 'none';
        }
    }
    
    enableWeb3Features() {
        document.getElementById('mint-coherence-nft').disabled = false;
        document.getElementById('stake-wisdom').disabled = false;
        
        // Add event listeners for Web3 actions
        document.getElementById('mint-coherence-nft').onclick = () => this.mintCoherenceNFT();
        document.getElementById('stake-wisdom').onclick = () => this.stakeWisdomTokens();
        document.getElementById('export-chain').onclick = () => this.exportChainData();
    }
    
    disableWeb3Features() {
        document.getElementById('mint-coherence-nft').disabled = true;
        document.getElementById('stake-wisdom').disabled = true;
        
        // Remove event listeners
        document.getElementById('mint-coherence-nft').onclick = null;
        document.getElementById('stake-wisdom').onclick = null;
    }
    
    formatAddress(address) {
        return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;\n    }\n    \n    getNetworkName(networkId) {\n        const networks = {\n            1: 'Ethereum Mainnet',\n            3: 'Ropsten Testnet',\n            4: 'Rinkeby Testnet',\n            5: 'Goerli Testnet',\n            42: 'Kovan Testnet',\n            137: 'Polygon Mainnet',\n            80001: 'Polygon Mumbai',\n            1337: 'Local Network'\n        };\n        \n        return networks[networkId] || `Network ${networkId}`;\n    }\n    \n    showInstallPrompt() {\n        const connectButton = document.getElementById('connect-wallet');\n        connectButton.textContent = 'Install MetaMask';\n        connectButton.onclick = () => {\n            window.open('https://metamask.io/download/', '_blank');\n        };\n    }\n    \n    showError(message) {\n        // You could implement a toast notification system here\n        alert(message);\n    }\n    \n    async mintCoherenceNFT() {\n        if (!this.isConnected) {\n            this.showError('Please connect your wallet first');\n            return;\n        }\n        \n        try {\n            // Get current coherence metrics from dashboard\n            const metrics = this.getCurrentCoherenceMetrics();\n            \n            // Create NFT metadata\n            const metadata = {\n                name: `IPAI Coherence Profile #${Date.now()}`,\n                description: 'A unique NFT representing your Grounded Coherence Theory profile',\n                image: await this.generateCoherenceArt(metrics),\n                attributes: [\n                    {\n                        trait_type: 'Internal Consistency (Ψ)',\n                        value: metrics.psi,\n                        max_value: 1\n                    },\n                    {\n                        trait_type: 'Accumulated Wisdom (ρ)',\n                        value: metrics.rho,\n                        max_value: 1\n                    },\n                    {\n                        trait_type: 'Moral Activation (q)',\n                        value: metrics.q,\n                        max_value: 1\n                    },\n                    {\n                        trait_type: 'Social Belonging (f)',\n                        value: metrics.f,\n                        max_value: 1\n                    },\n                    {\n                        trait_type: 'Soul Echo',\n                        value: metrics.soul_echo,\n                        max_value: 1\n                    }\n                ],\n                coherence_data: metrics,\n                timestamp: new Date().toISOString(),\n                blockchain_hash: await this.getLatestBlockHash()\n            };\n            \n            // In a real implementation, you would:\n            // 1. Upload metadata to IPFS\n            // 2. Call the smart contract to mint the NFT\n            // 3. Sign the transaction\n            \n            // For demo purposes, we'll simulate the minting\n            await this.simulateNFTMinting(metadata);\n            \n            // Log to helical blockchain\n            await this.logToHelicalBlockchain('nft_mint', {\n                metadata: metadata,\n                wallet: this.account\n            });\n            \n            alert('Coherence NFT minted successfully! (Simulated)');\n            \n        } catch (error) {\n            console.error('Failed to mint NFT:', error);\n            this.showError('Failed to mint NFT. Please try again.');\n        }\n    }\n    \n    async stakeWisdomTokens() {\n        if (!this.isConnected) {\n            this.showError('Please connect your wallet first');\n            return;\n        }\n        \n        try {\n            // Get wisdom metrics\n            const wisdomScore = parseFloat(document.getElementById('rho-value').textContent);\n            const stakeAmount = Math.floor(wisdomScore * 100); // Convert to token amount\n            \n            // Simulate staking transaction\n            const stakingData = {\n                amount: stakeAmount,\n                wisdom_score: wisdomScore,\n                staker: this.account,\n                duration: '30 days', // Default staking period\n                expected_rewards: stakeAmount * 0.1, // 10% APY simulation\n                timestamp: new Date().toISOString()\n            };\n            \n            // In a real implementation, you would call the staking contract\n            await this.simulateStaking(stakingData);\n            \n            // Log to helical blockchain\n            await this.logToHelicalBlockchain('wisdom_stake', stakingData);\n            \n            alert(`Successfully staked ${stakeAmount} Wisdom Tokens! (Simulated)`);\n            \n        } catch (error) {\n            console.error('Failed to stake tokens:', error);\n            this.showError('Failed to stake tokens. Please try again.');\n        }\n    }\n    \n    async exportChainData() {\n        try {\n            // Get blockchain state from IPAI backend\n            const response = await fetch('http://localhost:8000/api/v1/blockchain/export');\n            \n            let chainData;\n            if (response.ok) {\n                chainData = await response.json();\n            } else {\n                // Generate simulated export data\n                chainData = await this.generateSimulatedExport();\n            }\n            \n            // Create downloadable file\n            const blob = new Blob([JSON.stringify(chainData, null, 2)], {\n                type: 'application/json'\n            });\n            \n            const url = URL.createObjectURL(blob);\n            const a = document.createElement('a');\n            a.href = url;\n            a.download = `ipai-helical-blockchain-${new Date().toISOString().split('T')[0]}.json`;\n            a.click();\n            \n            URL.revokeObjectURL(url);\n            \n            // Log export action\n            await this.logToHelicalBlockchain('chain_export', {\n                exported_by: this.account || 'anonymous',\n                export_size: blob.size,\n                timestamp: new Date().toISOString()\n            });\n            \n            console.log('✅ Chain data exported successfully');\n            \n        } catch (error) {\n            console.error('Failed to export chain data:', error);\n            this.showError('Failed to export chain data. Please try again.');\n        }\n    }\n    \n    getCurrentCoherenceMetrics() {\n        return {\n            psi: parseFloat(document.getElementById('psi-value').textContent),\n            rho: parseFloat(document.getElementById('rho-value').textContent),\n            q: parseFloat(document.getElementById('q-value').textContent),\n            f: parseFloat(document.getElementById('f-value').textContent),\n            soul_echo: parseFloat(document.getElementById('soul-echo').textContent)\n        };\n    }\n    \n    async generateCoherenceArt(metrics) {\n        // Generate a unique visual representation of coherence metrics\n        const canvas = document.createElement('canvas');\n        canvas.width = 400;\n        canvas.height = 400;\n        const ctx = canvas.getContext('2d');\n        \n        // Background\n        ctx.fillStyle = '#0f0f23';\n        ctx.fillRect(0, 0, canvas.width, canvas.height);\n        \n        // Draw helical pattern based on metrics\n        const centerX = canvas.width / 2;\n        const centerY = canvas.height / 2;\n        const radius = 100;\n        \n        // Psi - Internal Consistency (spiral)\n        ctx.strokeStyle = `rgba(99, 102, 241, ${metrics.psi})`;\n        ctx.lineWidth = 3;\n        ctx.beginPath();\n        for (let i = 0; i < 360; i++) {\n            const angle = (i * Math.PI) / 180;\n            const r = radius * (1 - i / 360) * metrics.psi;\n            const x = centerX + r * Math.cos(angle);\n            const y = centerY + r * Math.sin(angle);\n            if (i === 0) ctx.moveTo(x, y);\n            else ctx.lineTo(x, y);\n        }\n        ctx.stroke();\n        \n        // Rho - Accumulated Wisdom (circles)\n        ctx.strokeStyle = `rgba(6, 182, 212, ${metrics.rho})`;\n        for (let i = 0; i < 5; i++) {\n            ctx.beginPath();\n            ctx.arc(centerX, centerY, (i + 1) * 20 * metrics.rho, 0, 2 * Math.PI);\n            ctx.stroke();\n        }\n        \n        // Q - Moral Activation (rays)\n        ctx.strokeStyle = `rgba(16, 185, 129, ${metrics.q})`;\n        ctx.lineWidth = 2;\n        for (let i = 0; i < 8; i++) {\n            const angle = (i * Math.PI) / 4;\n            ctx.beginPath();\n            ctx.moveTo(centerX, centerY);\n            ctx.lineTo(\n                centerX + 80 * metrics.q * Math.cos(angle),\n                centerY + 80 * metrics.q * Math.sin(angle)\n            );\n            ctx.stroke();\n        }\n        \n        // F - Social Belonging (connected dots)\n        ctx.fillStyle = `rgba(245, 158, 11, ${metrics.f})`;\n        const points = [];\n        for (let i = 0; i < 6; i++) {\n            const angle = (i * Math.PI) / 3;\n            const x = centerX + 60 * Math.cos(angle);\n            const y = centerY + 60 * Math.sin(angle);\n            points.push({ x, y });\n            \n            ctx.beginPath();\n            ctx.arc(x, y, 5 * metrics.f, 0, 2 * Math.PI);\n            ctx.fill();\n        }\n        \n        // Connect points\n        ctx.strokeStyle = `rgba(245, 158, 11, ${metrics.f * 0.5})`;\n        ctx.lineWidth = 1;\n        for (let i = 0; i < points.length; i++) {\n            for (let j = i + 1; j < points.length; j++) {\n                ctx.beginPath();\n                ctx.moveTo(points[i].x, points[i].y);\n                ctx.lineTo(points[j].x, points[j].y);\n                ctx.stroke();\n            }\n        }\n        \n        // Soul Echo center\n        ctx.fillStyle = `rgba(255, 255, 255, ${metrics.soul_echo})`;\n        ctx.beginPath();\n        ctx.arc(centerX, centerY, 10 * metrics.soul_echo, 0, 2 * Math.PI);\n        ctx.fill();\n        \n        // Convert to data URL (base64)\n        return canvas.toDataURL();\n    }\n    \n    async simulateNFTMinting(metadata) {\n        // Simulate blockchain transaction delay\n        await new Promise(resolve => setTimeout(resolve, 2000));\n        \n        // Store in localStorage for demo\n        const nfts = JSON.parse(localStorage.getItem('ipai_nfts') || '[]');\n        nfts.push({\n            id: nfts.length + 1,\n            metadata: metadata,\n            minted_at: new Date().toISOString(),\n            owner: this.account\n        });\n        localStorage.setItem('ipai_nfts', JSON.stringify(nfts));\n        \n        console.log('✅ NFT minted (simulated):', metadata);\n    }\n    \n    async simulateStaking(stakingData) {\n        // Simulate staking transaction\n        await new Promise(resolve => setTimeout(resolve, 1500));\n        \n        // Store in localStorage for demo\n        const stakes = JSON.parse(localStorage.getItem('ipai_stakes') || '[]');\n        stakes.push({\n            id: stakes.length + 1,\n            ...stakingData,\n            status: 'active'\n        });\n        localStorage.setItem('ipai_stakes', JSON.stringify(stakes));\n        \n        console.log('✅ Tokens staked (simulated):', stakingData);\n    }\n    \n    async generateSimulatedExport() {\n        return {\n            export_info: {\n                timestamp: new Date().toISOString(),\n                version: '1.0.0',\n                network: 'IPAI Helical Testnet',\n                exported_by: this.account || 'anonymous'\n            },\n            blockchain_data: {\n                total_blocks: Math.floor(Math.random() * 100) + 50,\n                strands: 5,\n                cross_links: Math.floor(Math.random() * 20) + 10,\n                latest_height: Math.floor(Math.random() * 50) + 25\n            },\n            sample_blocks: [\n                {\n                    hash: '0x1234...abcd',\n                    strand: 'coherence',\n                    coherence_score: 0.75,\n                    timestamp: new Date().toISOString()\n                },\n                {\n                    hash: '0x5678...efgh',\n                    strand: 'wisdom',\n                    coherence_score: 0.82,\n                    timestamp: new Date().toISOString()\n                }\n            ],\n            nfts: JSON.parse(localStorage.getItem('ipai_nfts') || '[]'),\n            stakes: JSON.parse(localStorage.getItem('ipai_stakes') || '[]')\n        };\n    }\n    \n    async logToHelicalBlockchain(action, data) {\n        try {\n            // Log Web3 actions to the helical blockchain\n            if (window.dashboard) {\n                await window.dashboard.logInteractionToBlockchain(`web3_${action}`, {\n                    ...data,\n                    wallet: this.account,\n                    network: this.networkId\n                });\n            }\n        } catch (error) {\n            console.error('Failed to log to helical blockchain:', error);\n        }\n    }\n    \n    async getLatestBlockHash() {\n        try {\n            const response = await fetch('http://localhost:8000/api/v1/blockchain/latest');\n            if (response.ok) {\n                const data = await response.json();\n                return data.hash;\n            }\n        } catch (error) {\n            console.log('Using simulated block hash');\n        }\n        \n        // Return simulated hash\n        return '0x' + Math.random().toString(16).substring(2, 18);\n    }\n}\n\n// Initialize when DOM is loaded\ndocument.addEventListener('DOMContentLoaded', () => {\n    window.web3Integration = new Web3Integration();\n});