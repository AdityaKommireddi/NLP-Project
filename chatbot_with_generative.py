"""F1 2025 Setup Assistant Chatbot with ML Intent Classification + Generative Chat
Integrates:
  - DistilBERT for intent classification (91% accuracy)
  - TinyLlama for open-ended generative chat fallback
  - Excel-based setup retrieval
  - Dynamic setup adjustment with value tracking
"""


import os
import sys
import re
import pandas as pd
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


from intent_classifier import IntentClassifier
from generative_layer_rag import GenerativeChatRAG, FALLBACK_RESPONSES


import random



class F125ChatbotWithGenerative:
    def __init__(self, setups_file: str = None, model_path: str = None, enable_generative: bool = True):
        """
        Initialize the F1 2025 chatbot with ML intent classification + optional generative layer.
        
        Args:
            setups_file: Path to Excel file with setup data
            model_path: Path to trained intent classifier model
            enable_generative: Whether to load TinyLlama model for open-ended chat
        """
        # Setup file path
        if setups_file is None:
            setups_file = project_root / "data" / "raw" / "F125-Setups.xlsx"
        
        # Load setup data
        try:
            self.setups_df = pd.read_excel(setups_file)
            print(f"✅ Loaded {len(self.setups_df)} setups from {setups_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load setups file: {e}")
            self.setups_df = None
        
        # Initialize ML intent classifier
        if model_path is None:
            model_path = project_root / "models" / "intent_classifier"
        
        try:
            self.intent_classifier = IntentClassifier(model_path)
            print(f"✅ Loaded ML intent classifier from {model_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load ML classifier: {e}")
            print("   Falling back to keyword-based intent detection only")
            self.intent_classifier = None
        
        # Initialize generative chat layer (TinyLlama or similar)
        self.generative_chat = None
        if enable_generative:
            try:
                print("\n🤖 Initializing generative chat layer...")
                self.generative_chat = GenerativeChatRAG(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_4bit=False)
                if self.generative_chat.is_available():
                    print("✅ Generative chat ready!\n")
                else:
                    self.generative_chat = None
            except Exception as e:
                print(f"⚠️ Generative chat not available: {e}")
                self.generative_chat = None
        
        # Track last setup for feedback adjustments
        self.last_setup_dict = None  # Stores parsed setup as dict
        self.last_track = None
        
        # Setup adjustment mappings (delta values)
        self.UNDERSTEER_ADJUSTMENTS = {
            'FW': +2,    # Front Wing
            'RW': -1,    # Rear Wing
            'FARB': -1,  # Front Anti-Roll Bar
            'RARB': +1,  # Rear Anti-Roll Bar
            'FC': -0.2,  # Front Camber (more negative)
            'FRT': -0.3, # Front Right Tire Pressure
            'FLT': -0.3, # Front Left Tire Pressure
        }
        
        self.OVERSTEER_ADJUSTMENTS = {
            'FW': -1,
            'RW': +2,
            'FARB': +1,
            'RARB': -1,
            'RC': +0.2,   # Rear Camber (less negative)
            'RRT': -0.3,
            'RLT': -0.3,
            'Diff on': -3, # More open diff
        }
        
        self.TIRE_OVERHEAT_ADJUSTMENTS = {
            'FC': +0.3,
            'RC': +0.3,
            'FRT': +0.5,
            'FLT': +0.5,
            'RRT': +0.5,
            'RLT': +0.5,
        }
        
        self.BRAKE_LOCK_ADJUSTMENTS = {
            'Pressure': -5,
            'Bias': -2,
        }
        
        # Track names mapping (handle variations)
        self.track_aliases = {
            'monaco': 'Monaco',
            'monza': 'Monza',
            'italy': 'Monza',
            'silverstone': 'Silverstone',
            'spa': 'Belgium Spa',
            'belgium': 'Belgium Spa',
            'spa-francorchamps': 'Belgium Spa',
            'austria': 'Austria',
            'red bull ring': 'Austria',
            'brazil': 'Brazil',
            'interlagos': 'Brazil',
            'sao paulo': 'Brazil',
            'hungary': 'Hungary',
            'hungaroring': 'Hungary',
            'singapore': 'Singapore',
            'marina bay': 'Singapore',
            'bahrain': 'Bahrain',
            'sakhir': 'Bahrain',
            'cota': 'COTA',
            'austin': 'COTA',
            'usa': 'COTA',
            'america': 'COTA',
            'suzuka': 'Japan',
            'japan': 'Japan',
            'baku': 'Baku',
            'azerbaijan': 'Baku',
            'abu dhabi': 'Abu dhabi',
            'yas marina': 'Abu dhabi',
            'zandvoort': 'Netherlands',
            'netherlands': 'Netherlands',
            'jeddah': 'Saudi Arabia',
            'saudi arabia': 'Saudi Arabia',
            'miami': 'Miami',
            'imola': 'Imola',
            'emilia romagna': 'Imola',
            'canada': 'Canada',
            'montreal': 'Canada',
            'spain': 'Spain',
            'barcelona': 'Spain',
            'china': 'China',
            'shanghai': 'China',
            'qatar': 'Qatar',
            'lusail': 'Qatar',
            'mexico': 'Mexican',
            'mexico city': 'Mexican',
            'las vegas': 'Las Vegas',
            'vegas': 'Las Vegas',
            'australia': 'Australia',
            'melbourne': 'Australia',
        }
        
        # Component explanations
        self.component_explanations = {
            'front wing': {
                'description': 'Controls front downforce and front-end grip',
                'higher': 'More front downforce, better turn-in, more understeer tendency',
                'lower': 'Less drag, higher top speed, less front grip'
            },
            'rear wing': {
                'description': 'Controls rear downforce and stability',
                'higher': 'More rear grip, better stability, lower top speed',
                'lower': 'Less drag, higher top speed, potential oversteer'
            },
            'differential': {
                'description': 'Controls how power is distributed between rear wheels',
                'on_throttle': 'Higher = more locked under acceleration (stability vs agility)',
                'off_throttle': 'Higher = more locked when coasting (stability vs rotation)'
            },
            'anti-roll bar': {
                'description': 'Controls body roll and load transfer during cornering',
                'stiffer': 'Less body roll, sharper response, less mechanical grip',
                'softer': 'More body roll, better mechanical grip, slower response'
            },
            'suspension': {
                'description': 'Controls how the car reacts to bumps and load changes',
                'stiff': 'Better aero platform, less mechanical grip, harsher over bumps',
                'soft': 'More mechanical grip, worse aero platform, better over bumps'
            },
            'camber': {
                'description': 'Tire angle relative to vertical when viewed from front/rear',
                'more_negative': 'Better cornering grip, worse straight-line grip',
                'less_negative': 'Better straight-line grip, worse cornering grip'
            },
            'toe': {
                'description': 'Tire angle relative to centerline when viewed from above',
                'toe_in': 'More stable, slower turn-in, more tire wear',
                'toe_out': 'Sharper turn-in, less stable, more tire wear'
            },
            'tire pressure': {
                'description': 'Air pressure inside the tire',
                'higher': 'Lower rolling resistance, less grip, less wear',
                'lower': 'More grip, more tire wear, risk of overheating'
            },
            'ride height': {
                'description': 'Distance between car floor and ground',
                'lower': 'More downforce, risk of bottoming out',
                'higher': 'Less downforce, better over bumps, more stable'
            },
            'brake bias': {
                'description': 'Distribution of braking force between front and rear',
                'forward': 'More front brake, risk of front locking',
                'rearward': 'More rear brake, risk of rear locking/instability'
            }
        }
    
    def _detect_intent(self, message: str) -> tuple:
        """
        Detect user intent using ML classifier with keyword fallback.
        
        Returns:
            tuple: (intent, confidence)
        """
        message_lower = message.lower().strip()
        
        # Try ML classifier first
        if self.intent_classifier:
            try:
                intent, confidence = self.intent_classifier.predict(message)
                
                # Use ML prediction if confidence is reasonable
                if confidence > 0.35:
                    return intent, confidence
                
                # Low confidence - fall through to keyword backup
                print(f"   [Low ML confidence: {confidence:.2%}, trying keywords]")
            
            except Exception as e:
                print(f"   [ML prediction error: {e}, trying keywords]")
        
        # Keyword-based fallback for very low confidence or ML failure
        return self._keyword_fallback(message_lower), 0.0
    
    def _keyword_fallback(self, message_lower: str) -> str:
        """Keyword-based intent detection as fallback."""
        
        # Setup request patterns
        if any(word in message_lower for word in ['setup', 'settings', 'configuration', 'tune', 'setp']):
            return 'setup_request'
        
        # Track guide patterns
        if any(word in message_lower for word in ['how to drive', 'tips', 'guide', 'racing line', 'braking point']):
            return 'track_guide'
        
        # Track tour patterns
        if any(phrase in message_lower for phrase in ['all tracks', 'track list', 'circuits', 'calendar', 'tour']):
            return 'track_tour'
        
        # Component explanation patterns
        if any(word in message_lower for word in ['what is', 'explain', 'how does', 'what does']) and \
           any(comp in message_lower for comp in ['wing', 'differential', 'suspension', 'camber', 'toe', 'brake bias']):
            return 'explain_component'
        
        # Feedback patterns
        if 'understeer' in message_lower or 'pushing' in message_lower or 'wont turn' in message_lower:
            return 'feedback_understeer'
        
        if 'oversteer' in message_lower or 'spinning' in message_lower or 'rear' in message_lower and 'loose' in message_lower:
            return 'feedback_oversteer'
        
        if 'tire' in message_lower or 'tyre' in message_lower:
            if 'overheat' in message_lower or 'hot' in message_lower or 'blister' in message_lower:
                return 'feedback_tire_overheat'
            if 'wear' in message_lower or 'deg' in message_lower:
                return 'feedback_tire_wear'
        
        if 'unstable' in message_lower or 'balance' in message_lower:
            return 'feedback_balance'
        
        if 'bottom' in message_lower or 'floor' in message_lower and 'hit' in message_lower:
            return 'feedback_bottoming'
        
        if 'brake' in message_lower and ('lock' in message_lower or 'flat spot' in message_lower):
            return 'feedback_brake_lock'
        
        # Greetings
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
            return 'greeting'
        
        # Thanks
        if any(word in message_lower for word in ['thanks', 'thank you', 'cheers', 'appreciate']):
            return 'thanks'
        
        # Default to general question (can use generative model)
        return 'general_question'
    
    def _extract_track(self, message: str) -> str:
        """Extract track name from message."""
        message_lower = message.lower()
        
        for alias, track_name in self.track_aliases.items():
            if alias in message_lower:
                return track_name
        
        return None
    
    def _get_setup(self, track: str) -> dict:
        """Retrieve setup for specific track, robust to column name issues."""
        if self.setups_df is None:
            return None


        # Allow for possible column variants
        col_candidates = ['Track List', 'Track', 'TRACK LIST']
        found = False
        for col in self.setups_df.columns:
            for candidate in col_candidates:
                if col.strip().lower() == candidate.strip().lower():
                    track_col = col
                    found = True
                    break
            if found:
                break
        else:
            print("❌ Could not find a Track column in Excel!")
            return None


        # Find matching track in dataframe (using robust column)
        setup = self.setups_df[self.setups_df[track_col].str.contains(track, case=False, na=False)]


        if setup.empty:
            return None


        # Return first match as dictionary
        return setup.iloc[0].to_dict()
    
    def _parse_values(self, text: str, keys: list, separator: str = '|') -> dict:
        """
        Parse compound format strings like "FW=20 | RW=14" or "FS=28|RS=14|..."
        Returns dict instead of tuple for easier manipulation.
        """
        result = {}
        
        if not text or pd.isna(text):
            for key in keys:
                result[key] = 'N/A'
            return result
        
        # Split by separator and clean
        parts = [p.strip() for p in text.split(separator)]
        
        # Build key-value dictionary
        data = {}
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                data[key.strip().upper()] = value.strip()
        
        # Extract values for requested keys
        for key in keys:
            value = data.get(key.upper(), 'N/A')
            # Try to convert to float for numeric values
            try:
                result[key] = float(value)
            except:
                result[key] = value
        
        return result
    
    def _parse_setup_to_dict(self, setup: dict) -> dict:
        """Convert raw setup dict to flat key-value dict for easy adjustment."""
        flat = {}
        
        # Parse Aerodynamics
        aero = setup.get('Aerodynamics', '')
        flat.update(self._parse_values(aero, ['FW', 'RW']))
        
        # Parse Transmission
        trans = setup.get('Transmission', '')
        flat.update(self._parse_values(trans, ['Diff on', 'Diff off']))
        
        # Parse Suspension Geometry
        geom = setup.get('Suspension Geometry', '')
        flat.update(self._parse_values(geom, ['FC', 'RC', 'FTO', 'RTI'], separator='|'))
        
        # Parse Suspension
        susp = setup.get('Suspension', '')
        flat.update(self._parse_values(susp, ['FS', 'RS', 'FARB', 'RARB', 'FRH', 'RRH'], separator='|'))
        
        # Parse Brakes
        brakes = setup.get('Brakes', '')
        flat.update(self._parse_values(brakes, ['Bias', 'Pressure'], separator='|'))
        
        # Parse Tires
        tires = setup.get('Tyres(DRY)', '')
        flat.update(self._parse_values(tires, ['FRT', 'FLT', 'RRT', 'RLT'], separator='|'))
        
        return flat
    
    def _apply_adjustments(self, setup_dict: dict, adjustments: dict) -> tuple:
        """
        Apply adjustment deltas to setup.
        Returns (new_setup_dict, changes_list).
        """
        new_setup = setup_dict.copy()
        changes = []
        
        for key, delta in adjustments.items():
            if key in new_setup and new_setup[key] != 'N/A':
                try:
                    old_val = float(new_setup[key])
                    new_val = round(old_val + delta, 1)
                    new_val = max(0, new_val)  # Don't go negative
                    new_setup[key] = new_val
                    
                    # Format change description
                    key_name = self._get_readable_name(key)
                    changes.append(f"• {key_name}: {old_val} → {new_val}")
                except ValueError:
                    pass
        
        return new_setup, changes
    
    def _get_readable_name(self, key: str) -> str:
        """Convert abbreviated keys to readable names."""
        names = {
            'FW': 'Front Wing',
            'RW': 'Rear Wing',
            'FC': 'Front Camber',
            'RC': 'Rear Camber',
            'FTO': 'Front Toe',
            'RTI': 'Rear Toe',
            'FS': 'Front Suspension',
            'RS': 'Rear Suspension',
            'FARB': 'Front Anti-Roll Bar',
            'RARB': 'Rear Anti-Roll Bar',
            'FRH': 'Front Ride Height',
            'RRH': 'Rear Ride Height',
            'Bias': 'Brake Bias',
            'Pressure': 'Brake Pressure',
            'Diff on': 'On-Throttle Diff',
            'Diff off': 'Off-Throttle Diff',
            'FRT': 'Front Right Tire Pressure',
            'FLT': 'Front Left Tire Pressure',
            'RRT': 'Rear Right Tire Pressure',
            'RLT': 'Rear Left Tire Pressure',
        }
        return names.get(key, key)
    
    def _format_setup(self, setup_dict: dict, flat_setup: dict = None) -> str:
        """Format setup dictionary into readable output."""
        if setup_dict is None:
            return "Setup not found."
        
        # Get track name
        track_col_candidates = ['Track List', 'Track', 'TRACK LIST']
        track_name = None
        for col in track_col_candidates:
            if col in setup_dict:
                track_name = setup_dict[col]
                break
        
        if track_name is None:
            track_name = list(setup_dict.values())[0]
        
        # Use provided flat setup or parse fresh
        if flat_setup is None:
            flat_setup = self._parse_setup_to_dict(setup_dict)
        
        response = f"\n🏁 **{track_name} Setup**\n\n"
        
        # Aerodynamics
        response += "**🔹 Aerodynamics:**\n"
        response += f"  • Front Wing: {flat_setup.get('FW', 'N/A')}\n"
        response += f"  • Rear Wing: {flat_setup.get('RW', 'N/A')}\n\n"
        
        # Differential
        response += "**🔹 Differential:**\n"
        response += f"  • On Throttle: {flat_setup.get('Diff on', 'N/A')}%\n"
        response += f"  • Off Throttle: {flat_setup.get('Diff off', 'N/A')}%\n\n"
        
        # Suspension Geometry
        response += "**🔹 Suspension Geometry:**\n"
        response += f"  • Front Camber: {flat_setup.get('FC', 'N/A')}°\n"
        response += f"  • Rear Camber: {flat_setup.get('RC', 'N/A')}°\n"
        response += f"  • Front Toe: {flat_setup.get('FTO', 'N/A')}°\n"
        response += f"  • Rear Toe: {flat_setup.get('RTI', 'N/A')}°\n\n"
        
        # Suspension
        response += "**🔹 Suspension:**\n"
        response += f"  • Front Suspension: {flat_setup.get('FS', 'N/A')}\n"
        response += f"  • Rear Suspension: {flat_setup.get('RS', 'N/A')}\n"
        response += f"  • Front Anti-Roll Bar: {flat_setup.get('FARB', 'N/A')}\n"
        response += f"  • Rear Anti-Roll Bar: {flat_setup.get('RARB', 'N/A')}\n\n"
        
        # Ride Height
        response += "**🔹 Ride Height:**\n"
        response += f"  • Front: {flat_setup.get('FRH', 'N/A')}\n"
        response += f"  • Rear: {flat_setup.get('RRH', 'N/A')}\n\n"
        
        # Brakes
        response += "**🔹 Brakes:**\n"
        response += f"  • Brake Pressure: {flat_setup.get('Pressure', 'N/A')}%\n"
        response += f"  • Brake Bias: {flat_setup.get('Bias', 'N/A')}%\n\n"
        
        # Tires
        response += "**🔹 Tires:**\n"
        response += f"  • Front Right Pressure: {flat_setup.get('FRT', 'N/A')} psi\n"
        response += f"  • Front Left Pressure: {flat_setup.get('FLT', 'N/A')} psi\n"
        response += f"  • Rear Right Pressure: {flat_setup.get('RRT', 'N/A')} psi\n"
        response += f"  • Rear Left Pressure: {flat_setup.get('RLT', 'N/A')} psi\n"
        
        return response
    
    def _handle_setup_request(self, message: str) -> str:
        """Handle setup request intent and save for adjustments."""
        track = self._extract_track(message)
        
        if track is None:
            return ("I'd be happy to provide a setup! Which track are you looking for?\n\n"
                   "Available tracks: Monaco, Monza, Silverstone, Spa, Austria, Hungary, "
                   "Singapore, Bahrain, Suzuka, and more...")
        
        setup = self._get_setup(track)
        
        if setup is None:
            return f"Sorry, I don't have setup data for {track} yet. Try another track!"
        
        # Parse and save for future feedback
        self.last_setup_dict = self._parse_setup_to_dict(setup)
        self.last_track = track
        
        return self._format_setup(setup, self.last_setup_dict)
    
    def _handle_track_guide(self, message: str) -> str:
        """Handle track driving guide request."""
        track = self._extract_track(message)
        
        if track is None:
            return ("I can help with driving tips! Which track would you like guidance for?\n\n"
                   "Just ask something like: 'How to drive Spa' or 'Monaco racing tips'")
        
        response = f"\n🏎️ **{track} Driving Tips:**\n\n"
        response += "**Key Tips:**\n"
        response += "• Study the racing line and braking points\n"
        response += "• Be smooth with steering inputs\n"
        response += "• Manage tire temperatures throughout the stint\n"
        response += "• Trail brake into corners for better rotation\n"
        response += "• Focus on exit speed for better lap times\n\n"
        
        if 'tc' in message.lower() or 'traction' in message.lower() or 'assist' in message.lower():
            response += "\n**Driving Without Assists:**\n"
            response += "• Be gentle on throttle application\n"
            response += "• Modulate brake pressure to avoid locking\n"
            response += "• Anticipate car behavior changes\n"
            response += "• Practice consistency over outright speed\n"
        
        return response
    
    def _handle_track_tour(self, message: str) -> str:
        """Handle track tour/list request."""
        response = "\n🌍 **F1 2025 Calendar Tracks:**\n\n"
        
        tracks = [
            "🇲🇨 Monaco", "🇮🇹 Monza", "🇬🇧 Silverstone", "🇧🇪 Belgium Spa",
            "🇦🇹 Austria", "🇭🇺 Hungary", "🇸🇬 Singapore", "🇧🇭 Bahrain",
            "🇯🇵 Japan", "🇦🇿 Baku", "🇦🇪 Abu Dhabi", "🇳🇱 Netherlands",
            "🇸🇦 Saudi Arabia", "🇺🇸 Miami", "🇮🇹 Imola", "🇨🇦 Canada",
            "🇪🇸 Spain", "🇧🇷 Brazil", "🇺🇸 COTA", "🇲🇽 Mexican",
            "🇶🇦 Qatar", "🇺🇸 Las Vegas", "🇦🇺 Australia", "🇨🇳 China"
        ]
        
        for i, track in enumerate(tracks, 1):
            response += f"{i}. {track}\n"
        
        response += "\n💡 Ask for a setup or driving tips for any track!"
        
        return response
    
    def _handle_explain_component(self, message: str) -> str:
        """Handle component explanation request."""
        message_lower = message.lower()
        
        for comp_key, comp_data in self.component_explanations.items():
            if comp_key.replace(' ', '') in message_lower.replace(' ', ''):
                response = f"\n⚙️ **{comp_key.title()}**\n\n"
                response += f"{comp_data['description']}\n\n"
                
                if 'higher' in comp_data:
                    response += f"**Higher Setting:** {comp_data['higher']}\n"
                if 'lower' in comp_data:
                    response += f"**Lower Setting:** {comp_data['lower']}\n"
                
                if 'on_throttle' in comp_data:
                    response += f"**On-Throttle:** {comp_data['on_throttle']}\n"
                if 'off_throttle' in comp_data:
                    response += f"**Off-Throttle:** {comp_data['off_throttle']}\n"
                
                if 'stiffer' in comp_data:
                    response += f"**Stiffer:** {comp_data['stiffer']}\n"
                if 'softer' in comp_data:
                    response += f"**Softer:** {comp_data['softer']}\n"
                
                return response
        
        return ("I can explain various car components like:\n"
               "• Front Wing & Rear Wing\n"
               "• Differential (On/Off Throttle)\n"
               "• Anti-Roll Bars\n"
               "• Suspension\n"
               "• Camber & Toe\n"
               "• Tire Pressure\n"
               "• Ride Height\n"
               "• Brake Bias\n\n"
               "What would you like to know about?")
    
    def _handle_feedback_understeer(self, message: str) -> str:
        """Handle understeer feedback with dynamic setup adjustment."""
        if self.last_setup_dict is None:
            return "Please ask for a setup first (e.g., 'setup for Monaco') before giving feedback."
        
        # Apply adjustments
        new_setup, changes = self._apply_adjustments(self.last_setup_dict, self.UNDERSTEER_ADJUSTMENTS)
        
        # Update stored setup for chaining
        self.last_setup_dict = new_setup
        
        # Format response
        response = "\n🔧 **Setup Changes to Reduce Understeer:**\n"
        response += "=" * 50 + "\n\n"
        response += "\n".join(changes)
        response += "\n\n💡 Try this adjusted setup! Let me know if you need more changes."
        
        return response
    
    def _handle_feedback_oversteer(self, message: str) -> str:
        """Handle oversteer feedback with dynamic setup adjustment."""
        if self.last_setup_dict is None:
            return "Please ask for a setup first (e.g., 'setup for Spa') before giving feedback."
        
        new_setup, changes = self._apply_adjustments(self.last_setup_dict, self.OVERSTEER_ADJUSTMENTS)
        self.last_setup_dict = new_setup
        
        response = "\n🔧 **Setup Changes to Reduce Oversteer:**\n"
        response += "=" * 50 + "\n\n"
        response += "\n".join(changes)
        response += "\n\n💡 Try this adjusted setup! Let me know if you need more changes."
        
        return response
    
    def _handle_feedback_tire_overheat(self, message: str) -> str:
        """Handle tire overheating feedback with dynamic adjustment."""
        if self.last_setup_dict is None:
            return "Please ask for a setup first before giving feedback."
        
        new_setup, changes = self._apply_adjustments(self.last_setup_dict, self.TIRE_OVERHEAT_ADJUSTMENTS)
        self.last_setup_dict = new_setup
        
        response = "\n🔧 **Setup Changes to Reduce Tire Overheating:**\n"
        response += "=" * 50 + "\n\n"
        response += "\n".join(changes)
        response += "\n\n💡 Try this adjusted setup! Let me know if you need more changes."
        
        return response
    
    def _handle_feedback_tire_wear(self, message: str) -> str:
        """Handle tire wear feedback."""
        response = "\n🔧 **Tire Wear Management:**\n\n"
        response += "**Setup Changes:**\n"
        response += "• ⬆️ Increase tire pressure (reduces contact patch)\n"
        response += "• Reduce camber angles\n"
        response += "• Minimize toe settings\n"
        response += "• Soften suspension for better tire preservation\n\n"
        response += "**Driving Technique:**\n"
        response += "• Avoid excessive wheelspin\n"
        response += "• Smooth inputs (steering, throttle, brake)\n"
        response += "• Lift and coast into corners\n"
        response += "• Avoid aggressive curb usage\n"
        response += "• Manage tire temps (not too hot or cold)\n"
        
        return response
    
    def _handle_feedback_balance(self, message: str) -> str:
        """Handle balance/stability feedback."""
        response = "\n🔧 **Improving Car Balance:**\n\n"
        response += "**For Better Stability:**\n"
        response += "• Check aero balance (front/rear wing ratio)\n"
        response += "• Match anti-roll bar stiffness front-to-rear\n"
        response += "• Ensure suspension isn't too stiff\n"
        response += "• Check ride height (not too low)\n"
        response += "• Review differential settings\n\n"
        response += "**Diagnosis:**\n"
        response += "• Unpredictable = check suspension damping\n"
        response += "• Nervous = soften anti-roll bars\n"
        response += "• Both ends sliding = reduce overall grip demands\n"
        
        return response
    
    def _handle_feedback_bottoming(self, message: str) -> str:
        """Handle bottoming out feedback."""
        response = "\n🔧 **Bottoming Out Fixes:**\n\n"
        response += "**Setup Changes:**\n"
        response += "• ⬆️ Increase ride height (front and rear)\n"
        response += "• ⬆️ Stiffen suspension (resist compression)\n"
        response += "• Adjust dampers for bump resistance\n\n"
        response += "**Track-Specific:**\n"
        response += "• High-speed tracks need higher ride height\n"
        response += "• Bumpy tracks need softer suspension\n"
        response += "• Balance downforce loss vs. bottoming risk\n"
        
        return response
    
    def _handle_feedback_brake_lock(self, message: str) -> str:
        """Handle brake locking feedback with dynamic adjustment."""
        if self.last_setup_dict is None:
            return "Please ask for a setup first before giving feedback."
        
        new_setup, changes = self._apply_adjustments(self.last_setup_dict, self.BRAKE_LOCK_ADJUSTMENTS)
        self.last_setup_dict = new_setup
        
        response = "\n🔧 **Setup Changes to Reduce Brake Locking:**\n"
        response += "=" * 50 + "\n\n"
        response += "\n".join(changes)
        response += "\n\n💡 Try this adjusted setup! Let me know if you need more changes."
        
        return response
    
    def _handle_general_question(self, message: str) -> str:
        """
        Handle general questions.
        Uses generative LLM if available, otherwise fallback.
        """
        # Try generative model first
        if self.generative_chat and self.generative_chat.is_available():
            print("   [Using TinyLlama generative model with RAG for response]")
            reply = self.generative_chat.generate_reply(message, max_new_tokens=150, temperature=0.7, use_rag=True)
            if reply:
                return reply
        
        # Fallback to predefined responses
        return random.choice(FALLBACK_RESPONSES.get('general_question', FALLBACK_RESPONSES['general_question']))
    
    def _handle_greeting(self, message: str) -> str:
        """Handle greetings."""
        return ("👋 Hey there! Welcome to PitStop AI!\n\n"
               "I'm here to help with F1 25 setups, driving tips, and car setup advice.\n\n"
               "What can I help you with today?")
    
    def _handle_thanks(self, message: str) -> str:
        """Handle thank you messages."""
        return ("You're welcome! 😊\n\n"
               "Happy racing! Let me know if you need anything else. 🏁")
    
    def chat(self, message: str) -> str:
        """
        Main chat interface - process message and return response.
        """
        # Detect intent using ML + keyword fallback
        intent, confidence = self._detect_intent(message)
        
        # Log detection result
        if confidence > 0:
            print(f"🎯 Intent: {intent} (DistilBERT confidence: {confidence:.1%})")
        else:
            print(f"🎯 Intent: {intent} (keyword fallback)")
        
        # Route to appropriate handler
        handlers = {
            'setup_request': self._handle_setup_request,
            'track_guide': self._handle_track_guide,
            'track_tour': self._handle_track_tour,
            'explain_component': self._handle_explain_component,
            'feedback_understeer': self._handle_feedback_understeer,
            'feedback_oversteer': self._handle_feedback_oversteer,
            'feedback_tire_overheat': self._handle_feedback_tire_overheat,
            'feedback_tire_wear': self._handle_feedback_tire_wear,
            'feedback_balance': self._handle_feedback_balance,
            'feedback_bottoming': self._handle_feedback_bottoming,
            'feedback_brake_lock': self._handle_feedback_brake_lock,
            'general_question': self._handle_general_question,
            'greeting': self._handle_greeting,
            'thanks': self._handle_thanks,
        }
        
        handler = handlers.get(intent, self._handle_general_question)
        return handler(message)



def main():
    """Interactive chatbot session."""
    print("\n" + "="*70)
    print("🏁 PITSTOP AI - F1 2025 Setup Assistant (with Generative Chat)")
    print("="*70)
    print("Models: DistilBERT (intent) + TinyLlama (generative fallback)")
    print("="*70)
    
    # Initialize chatbot with generative layer
    bot = F125ChatbotWithGenerative(enable_generative=True)
    
    print("\n💬 Chat started! (type 'quit' or 'exit' to end)\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\nBot: Thanks for using PitStop AI! Happy racing! 🏁\n")
                break
            
            # Get bot response
            response = bot.chat(user_input)
            print(f"\nBot: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye! 🏁\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")



if __name__ == "__main__":
    main()
