import unittest
from unittest.mock import patch

from amap_client import find_balanced_center
from app_v2 import _validate_participants, app, calculate_routes


class MultiParticipantTest(unittest.TestCase):
    def setUp(self):
        self.participants = [
            {"id": "p1", "name": "甲", "location": {"lng": 116.0, "lat": 39.0, "name": "A"}, "preference": "transit"},
            {"id": "p2", "name": "乙", "location": {"lng": 117.0, "lat": 40.0, "name": "B"}, "preference": "driving"},
            {"id": "p3", "name": "丙", "location": {"lng": 118.0, "lat": 41.0, "name": "C"}, "preference": "walking"},
        ]

    def test_validates_two_to_eight_participants(self):
        self.assertEqual(_validate_participants(self.participants), self.participants)
        with self.assertRaisesRegex(ValueError, "2-8"):
            _validate_participants(self.participants[:1])
        with self.assertRaisesRegex(ValueError, "2-8"):
            _validate_participants(self.participants * 3)

    def test_balanced_center_uses_all_participants(self):
        result = find_balanced_center(self.participants)
        self.assertEqual(result["midpoint"], {"lng": 117.0, "lat": 40.0})
        self.assertGreaterEqual(result["suggested_search_radius_m"], 500)

    def test_uses_current_deepseek_model(self):
        import app_v2
        self.assertEqual(app_v2.DEEPSEEK_MODEL, "deepseek-v4-flash")

    def test_search_api_rejects_legacy_two_location_payload(self):
        response = app.test_client().post("/api/v2/search", json={
            "location_a": self.participants[0]["location"],
            "location_b": self.participants[1]["location"],
            "query": "餐厅",
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("2-8", response.get_json()["error"])

    @patch("app_v2.amap_get_best_route")
    def test_calculate_routes_aggregates_every_participant(self, get_route):
        get_route.side_effect = [
            {"success": True, "mode": "transit", "duration_minutes": 10, "duration_text": "10分钟"},
            {"success": True, "mode": "driving", "duration_minutes": 20, "duration_text": "20分钟"},
            {"success": True, "mode": "walking", "duration_minutes": 35, "duration_text": "35分钟"},
        ]
        pois = [{"name": "会合点", "lng": 116.5, "lat": 39.5, "rating": 4.5}]

        result = calculate_routes(pois, self.participants, city="北京")

        self.assertEqual(len(result[0]["routes"]), 3)
        self.assertEqual(result[0]["total_time_minutes"], 65)
        self.assertEqual(result[0]["time_range_minutes"], 25)


if __name__ == "__main__":
    unittest.main()
