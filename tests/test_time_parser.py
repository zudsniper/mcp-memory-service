"""Test the time_parser module."""
import pytest
from datetime import datetime, timedelta
import time

from mcp_memory_service.utils.time_parser import (
    parse_time_expression,
    extract_time_expression,
    get_time_of_day_range,
    get_last_period_range,
    get_this_period_range,
    get_month_range,
    get_named_period_range
)

def test_parse_relative_days():
    """Test parsing relative day expressions."""
    # Test 'X days ago'
    start_ts, end_ts = parse_time_expression("3 days ago")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    # The date should be 3 days ago
    expected_date = datetime.now().date() - timedelta(days=3)
    assert start_dt.date() == expected_date
    assert end_dt.date() == expected_date
    
    # The time range should cover the full day
    assert start_dt.hour == 0 and start_dt.minute == 0 and start_dt.second == 0
    assert end_dt.hour == 23 and end_dt.minute == 59 and end_dt.second == 59

def test_parse_yesterday_today():
    """Test parsing 'yesterday' and 'today'."""
    # Test 'yesterday'
    start_ts, end_ts = parse_time_expression("yesterday")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    expected_date = datetime.now().date() - timedelta(days=1)
    assert start_dt.date() == expected_date
    assert end_dt.date() == expected_date
    
    # Test 'today'
    start_ts, end_ts = parse_time_expression("today")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    expected_date = datetime.now().date()
    assert start_dt.date() == expected_date
    assert end_dt.date() == expected_date

def test_parse_relative_weeks():
    """Test parsing 'X weeks ago'."""
    start_ts, end_ts = parse_time_expression("2 weeks ago")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    # Should be a week starting 2 weeks ago (Monday to Sunday)
    today = datetime.now().date()
    days_since_monday = today.weekday()
    expected_monday = today - timedelta(days=days_since_monday) - timedelta(weeks=2)
    expected_sunday = expected_monday + timedelta(days=6)
    
    assert start_dt.date() == expected_monday
    assert end_dt.date() == expected_sunday

def test_parse_relative_months():
    """Test parsing 'X months ago'."""
    start_ts, end_ts = parse_time_expression("1 month ago")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    # Should be the previous calendar month
    now = datetime.now()
    
    if now.month == 1:  # January
        expected_month = 12  # December
        expected_year = now.year - 1
    else:
        expected_month = now.month - 1
        expected_year = now.year
    
    assert start_dt.year == expected_year
    assert start_dt.month == expected_month
    assert start_dt.day == 1  # First day of month
    
    # Last day of month varies
    if expected_month == 12:  # December
        expected_last_day = datetime(expected_year + 1, 1, 1) - timedelta(days=1)
    else:
        expected_last_day = datetime(expected_year, expected_month + 1, 1) - timedelta(days=1)
    
    assert end_dt.date() == expected_last_day.date()

def test_parse_relative_years():
    """Test parsing 'X years ago'."""
    start_ts, end_ts = parse_time_expression("2 years ago")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    expected_year = datetime.now().year - 2
    
    assert start_dt.year == expected_year
    assert start_dt.month == 1  # January
    assert start_dt.day == 1    # 1st
    
    assert end_dt.year == expected_year
    assert end_dt.month == 12   # December
    assert end_dt.day == 31     # 31st

def test_parse_last_period():
    """Test parsing 'last X' expressions."""
    # Test 'last week'
    start_ts, end_ts = parse_time_expression("last week")
    assert start_ts is not None and end_ts is not None
    
    # Test 'last month'
    start_ts, end_ts = parse_time_expression("last month")
    assert start_ts is not None and end_ts is not None
    
    # Test 'last year'
    start_ts, end_ts = parse_time_expression("last year")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    expected_year = datetime.now().year - 1
    assert start_dt.year == expected_year
    assert start_dt.month == 1   # January
    assert start_dt.day == 1     # 1st
    
    assert end_dt.year == expected_year
    assert end_dt.month == 12    # December
    assert end_dt.day == 31      # 31st

def test_parse_this_period():
    """Test parsing 'this X' expressions."""
    # Test 'this week'
    start_ts, end_ts = parse_time_expression("this week")
    assert start_ts is not None and end_ts is not None
    
    # Should be current week (Monday to Sunday)
    today = datetime.now().date()
    days_since_monday = today.weekday()
    expected_monday = today - timedelta(days=days_since_monday)
    expected_sunday = expected_monday + timedelta(days=6)
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    assert start_dt.date() == expected_monday
    assert end_dt.date() == expected_sunday
    
    # Test 'this month'
    start_ts, end_ts = parse_time_expression("this month")
    assert start_ts is not None and end_ts is not None
    
    # Should be current calendar month
    now = datetime.now()
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    assert start_dt.year == now.year
    assert start_dt.month == now.month
    assert start_dt.day == 1  # First day of month

def test_parse_month_name():
    """Test parsing month names."""
    # Test with a past month
    start_ts, end_ts = parse_time_expression("january")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    # Should be January of the current or previous year
    now = datetime.now()
    expected_year = now.year if now.month > 1 else now.year - 1
    
    assert start_dt.year == expected_year
    assert start_dt.month == 1  # January
    assert start_dt.day == 1    # 1st day
    
    # Last day of January
    assert end_dt.year == expected_year
    assert end_dt.month == 1   # January
    assert end_dt.day == 31    # 31st

def test_parse_specific_date():
    """Test parsing specific date formats."""
    # Test MM/DD/YYYY format
    start_ts, end_ts = parse_time_expression("12/25/2023")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    assert start_dt.year == 2023
    assert start_dt.month == 12  # December
    assert start_dt.day == 25    # 25th
    
    assert end_dt.year == 2023
    assert end_dt.month == 12
    assert end_dt.day == 25
    assert end_dt.hour == 23
    assert end_dt.minute == 59
    assert end_dt.second == 59
    
    # Test MM/DD format (current year)
    start_ts, end_ts = parse_time_expression("7/4")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    
    now = datetime.now()
    expected_year = now.year
    
    assert start_dt.year == expected_year
    assert start_dt.month == 7  # July
    assert start_dt.day == 4    # 4th

def test_parse_named_period():
    """Test parsing named periods like holidays."""
    # Test Christmas
    start_ts, end_ts = parse_time_expression("christmas")
    assert start_ts is not None and end_ts is not None
    
    # Test Summer
    start_ts, end_ts = parse_time_expression("summer")
    assert start_ts is not None and end_ts is not None
    
    # Test Spring
    start_ts, end_ts = parse_time_expression("spring")
    assert start_ts is not None and end_ts is not None

def test_parse_time_of_day():
    """Test parsing time of day expressions."""
    # Test 'yesterday morning'
    start_ts, end_ts = parse_time_expression("yesterday morning")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    expected_date = datetime.now().date() - timedelta(days=1)
    assert start_dt.date() == expected_date
    assert end_dt.date() == expected_date
    
    # Morning should be roughly 5AM-12PM
    assert start_dt.hour >= 5
    assert end_dt.hour <= 12

def test_extract_time_expression():
    """Test extracting time expressions from queries."""
    # Simple case with just a time expression
    query, (start_ts, end_ts) = extract_time_expression("yesterday")
    assert query.strip() == ""
    assert start_ts is not None and end_ts is not None
    
    # Mixed query with time expression
    query, (start_ts, end_ts) = extract_time_expression("find information about databases from 2 months ago")
    assert query.strip() == "find information about databases from"
    assert start_ts is not None and end_ts is not None
    
    # No time expression
    query, (start_ts, end_ts) = extract_time_expression("find information about databases")
    assert query.strip() == "find information about databases"
    assert start_ts is None and end_ts is None

def test_date_range():
    """Test parsing date ranges."""
    # Test 'between X and Y'
    start_ts, end_ts = parse_time_expression("between January and March")
    assert start_ts is not None and end_ts is not None
    
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    
    now = datetime.now()
    expected_year = now.year
    
    # If the current month is after March, the range refers to this year
    # If the current month is before January, the range refers to last year
    if now.month < 1:
        expected_year -= 1
    
    assert start_dt.year == expected_year
    assert start_dt.month == 1  # January
    
    assert end_dt.year == expected_year
    assert end_dt.month == 3   # March
    assert end_dt.day == 31    # Last day of March

def test_complex_query_with_time():
    """Test extracting time expressions from complex queries."""
    complex_query = "I need information about databases that I saved last week and details about API architecture from 3 months ago"
    
    # This should extract both time expressions and return the first one
    query, (start_ts, end_ts) = extract_time_expression(complex_query)
    
    # The cleaned query should remove the time expressions
    assert "last week" not in query
    assert "3 months ago" not in query
    assert start_ts is not None and end_ts is not None