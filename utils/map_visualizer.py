#!/usr/bin/env python3
"""
Map Stitching Visualization System

Creates visual representations of the stitched world map showing
connections between different areas, routes, towns, and buildings.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from utils.map_stitcher import MapStitcher, WarpConnection, MapArea
from utils.map_formatter import format_map_for_display, get_symbol_legend

logger = logging.getLogger(__name__)

class MapVisualizer:
    """Visualizes stitched map connections and layouts"""
    
    def __init__(self, map_stitcher: MapStitcher):
        self.stitcher = map_stitcher
    
    def generate_world_map_summary(self) -> str:
        """Generate a text summary of the entire stitched world"""
        stats = self.stitcher.get_stats()
        lines = [
            "=== STITCHED WORLD MAP SUMMARY ===",
            "",
            f"📍 Total Areas Discovered: {stats['total_areas']}",
            f"   🏠 Indoor Areas: {stats['indoor_areas']}",
            f"   🌍 Outdoor Areas: {stats['outdoor_areas']}",
            f"",
            f"🔗 Total Connections: {stats['total_connections']}",
        ]
        
        # Show warp type breakdown
        if stats['warp_types']:
            lines.append("   Warp Types:")
            for warp_type, count in stats['warp_types'].items():
                lines.append(f"     {warp_type}: {count}")
        
        lines.extend([
            "",
            f"⭐ Most Visited: {stats.get('most_visited', 'None')}",
            ""
        ])
        
        return "\n".join(lines)
    
    def generate_area_connections_map(self, focus_area_id: Optional[int] = None) -> str:
        """Generate a connection map showing how areas link together"""
        lines = ["=== AREA CONNECTIONS MAP ===", ""]
        
        if focus_area_id:
            # Show connections for a specific area
            area = self.stitcher.map_areas.get(focus_area_id)
            if not area:
                return "Area not found"
                
            lines.append(f"🎯 FOCUS: {area.location_name} (ID: {focus_area_id:04X})")
            lines.append("")
            
            connections = self.stitcher.get_connected_areas(focus_area_id)
            if connections:
                lines.append("Connected Areas:")
                for to_id, to_name, direction in connections:
                    direction_symbol = self._get_direction_symbol(direction)
                    lines.append(f"  {direction_symbol} {to_name} (ID: {to_id:04X})")
            else:
                lines.append("  No connections found")
        else:
            # Show all areas and their connection counts
            lines.append("All Areas (with connection counts):")
            lines.append("")
            
            for area_id, area in self.stitcher.map_areas.items():
                connections = self.stitcher.get_connected_areas(area_id)
                connection_count = len(connections)
                
                area_type = "🏠" if area.location_name and "HOUSE" in area.location_name.upper() else "🌍"
                lines.append(f"{area_type} {area.location_name} ({area_id:04X}) - {connection_count} connections")
                
                if connections:
                    for to_id, to_name, direction in connections[:3]:  # Show first 3
                        direction_symbol = self._get_direction_symbol(direction)
                        lines.append(f"    {direction_symbol} {to_name}")
                    if len(connections) > 3:
                        lines.append(f"    ... and {len(connections) - 3} more")
        
        return "\n".join(lines)
    
    def generate_route_network_map(self) -> str:
        """Generate a network view of route connections"""
        lines = ["=== ROUTE NETWORK MAP ===", ""]
        
        # Group areas by type
        routes = []
        towns = []
        buildings = []
        
        for area_id, area in self.stitcher.map_areas.items():
            name = area.location_name.upper() if area.location_name else "UNKNOWN"
            if "ROUTE" in name:
                routes.append((area_id, area))
            elif any(keyword in name for keyword in ["TOWN", "CITY"]):
                towns.append((area_id, area))
            elif any(keyword in name for keyword in ["HOUSE", "ROOM", "CENTER", "MART", "GYM"]):
                buildings.append((area_id, area))
        
        if routes:
            lines.append("🛤️  ROUTES:")
            for area_id, area in sorted(routes, key=lambda x: x[1].location_name):
                connections = self.stitcher.get_connected_areas(area_id)
                connection_names = [name for _, name, _ in connections]
                lines.append(f"  {area.location_name} → {', '.join(connection_names[:3])}")
        
        if towns:
            lines.append("")
            lines.append("🏘️  TOWNS & CITIES:")
            for area_id, area in sorted(towns, key=lambda x: x[1].location_name):
                connections = self.stitcher.get_connected_areas(area_id)
                route_connections = [name for _, name, _ in connections if "ROUTE" in name.upper()]
                if route_connections:
                    lines.append(f"  {area.location_name} ↔ {', '.join(route_connections)}")
                else:
                    lines.append(f"  {area.location_name} (isolated)")
        
        if buildings:
            lines.append("")
            lines.append("🏢 BUILDINGS:")
            building_count_by_area = {}
            for area_id, area in buildings:
                # Group buildings by their location area
                connections = self.stitcher.get_connected_areas(area_id)
                parent_areas = [name for _, name, _ in connections 
                              if not any(kw in name.upper() for kw in ["HOUSE", "ROOM", "CENTER"])]
                parent = parent_areas[0] if parent_areas else "Unknown"
                if parent not in building_count_by_area:
                    building_count_by_area[parent] = []
                building_count_by_area[parent].append(area.location_name)
            
            for parent, buildings_list in building_count_by_area.items():
                lines.append(f"  {parent}: {len(buildings_list)} buildings")
                for building in buildings_list[:3]:  # Show first 3
                    lines.append(f"    • {building}")
                if len(buildings_list) > 3:
                    lines.append(f"    • ... and {len(buildings_list) - 3} more")
        
        return "\n".join(lines)
    
    def generate_warp_details_report(self) -> str:
        """Generate detailed warp connection information"""
        lines = ["=== WARP CONNECTIONS DETAILS ===", ""]
        
        # Group connections by type
        warp_by_type = {}
        for conn in self.stitcher.warp_connections:
            if conn.warp_type not in warp_by_type:
                warp_by_type[conn.warp_type] = []
            warp_by_type[conn.warp_type].append(conn)
        
        for warp_type, connections in warp_by_type.items():
            type_symbol = self._get_warp_type_symbol(warp_type)
            lines.append(f"{type_symbol} {warp_type.upper()} CONNECTIONS ({len(connections)}):")
            
            for conn in connections[:10]:  # Show first 10 of each type
                from_area = self.stitcher.map_areas.get(conn.from_map_id)
                to_area = self.stitcher.map_areas.get(conn.to_map_id)
                
                if from_area and to_area:
                    direction_symbol = self._get_direction_symbol(conn.direction)
                    lines.append(f"  {direction_symbol} {from_area.location_name} → {to_area.location_name}")
                    lines.append(f"    Position: ({conn.from_position[0]}, {conn.from_position[1]}) → ({conn.to_position[0]}, {conn.to_position[1]})")
            
            if len(connections) > 10:
                lines.append(f"  ... and {len(connections) - 10} more {warp_type} connections")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_navigation_hints(self, current_area_id: int, target_area_name: str) -> str:
        """Generate navigation hints to reach a target area"""
        lines = [f"=== NAVIGATION: TO {target_area_name.upper()} ===", ""]
        
        current_area = self.stitcher.map_areas.get(current_area_id)
        if not current_area:
            return "Current area not found in stitched map"
        
        # Find target area
        target_area = None
        target_id = None
        for area_id, area in self.stitcher.map_areas.items():
            if area.location_name and target_area_name.upper() in area.location_name.upper():
                target_area = area
                target_id = area_id
                break
        
        if not target_area:
            lines.append(f"❌ Target area '{target_area_name}' not found in discovered areas")
            lines.append("")
            lines.append("Available areas:")
            for area in self.stitcher.map_areas.values():
                lines.append(f"  • {area.location_name}")
            return "\n".join(lines)
        
        lines.append(f"📍 Current: {current_area.location_name}")
        lines.append(f"🎯 Target: {target_area.location_name}")
        lines.append("")
        
        # Simple pathfinding - direct connections first
        direct_connections = self.stitcher.get_connected_areas(current_area_id)
        direct_targets = [conn for conn in direct_connections if conn[0] == target_id]
        
        if direct_targets:
            conn = direct_targets[0]
            direction_symbol = self._get_direction_symbol(conn[2])
            lines.append(f"🔗 Direct connection available!")
            lines.append(f"   {direction_symbol} Go {conn[2]} to reach {target_area.location_name}")
        else:
            # Look for paths through connected areas
            paths = self._find_simple_paths(current_area_id, target_id, max_depth=3)
            if paths:
                lines.append("🗺️  Possible routes:")
                for i, path in enumerate(paths[:3]):  # Show first 3 paths
                    route_description = self._describe_path(path)
                    lines.append(f"   Route {i+1}: {route_description}")
            else:
                lines.append("❓ No known path found (areas may not be connected yet)")
        
        return "\n".join(lines)
    
    def _find_simple_paths(self, start_id: int, target_id: int, max_depth: int = 3) -> List[List[int]]:
        """Find simple paths between areas (basic BFS)"""
        if start_id == target_id:
            return [[start_id]]
        
        visited = set()
        queue = [(start_id, [start_id])]
        paths = []
        
        while queue and len(paths) < 5:  # Limit to 5 paths
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
                
            if current_id in visited:
                continue
                
            visited.add(current_id)
            
            connections = self.stitcher.get_connected_areas(current_id)
            for next_id, _, _ in connections:
                if next_id == target_id:
                    paths.append(path + [next_id])
                elif next_id not in visited and len(path) < max_depth:
                    queue.append((next_id, path + [next_id]))
        
        return paths
    
    def _describe_path(self, path: List[int]) -> str:
        """Describe a path as a sequence of area names"""
        if len(path) < 2:
            return "No path"
        
        descriptions = []
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]
            
            from_area = self.stitcher.map_areas.get(from_id)
            to_area = self.stitcher.map_areas.get(to_id)
            
            if from_area and to_area:
                # Find the connection direction
                connections = self.stitcher.get_connected_areas(from_id)
                direction = "→"
                for conn_id, _, conn_dir in connections:
                    if conn_id == to_id:
                        direction = self._get_direction_symbol(conn_dir)
                        break
                
                descriptions.append(f"{from_area.location_name} {direction} {to_area.location_name}")
        
        return " → ".join(descriptions)
    
    def _get_direction_symbol(self, direction: str) -> str:
        """Get symbol for direction"""
        symbols = {
            "north": "⬆️", "south": "⬇️", "east": "➡️", "west": "⬅️",
            "up": "🔼", "down": "🔽", "northeast": "↗️", "northwest": "↖️",
            "southeast": "↘️", "southwest": "↙️"
        }
        return symbols.get(direction.lower(), "🔄")
    
    def _get_warp_type_symbol(self, warp_type: str) -> str:
        """Get symbol for warp type"""
        symbols = {
            "door": "🚪", "stairs": "🪜", "warp": "🌀", 
            "route_transition": "🛤️", "exit": "🚪"
        }
        return symbols.get(warp_type, "🔗")
    
    def generate_complete_world_overview(self) -> str:
        """Generate a complete overview combining all visualization types"""
        lines = [
            "=" * 60,
            "           POKEMON EMERALD WORLD MAP",
            "                (Stitched View)",
            "=" * 60,
            "",
            self.generate_world_map_summary(),
            "",
            self.generate_route_network_map(),
            "",
            self.generate_area_connections_map(),
            "",
            self.generate_warp_details_report(),
            "",
            "=" * 60,
            f"Generated from {len(self.stitcher.map_areas)} discovered areas",
            "=" * 60
        ]
        
        return "\n".join(lines)

def create_map_visualizer(memory_reader) -> MapVisualizer:
    """Create a map visualizer from a memory reader's stitcher"""
    if hasattr(memory_reader, '_map_stitcher'):
        return MapVisualizer(memory_reader._map_stitcher)
    else:
        # Create a standalone stitcher
        stitcher = MapStitcher()
        return MapVisualizer(stitcher)