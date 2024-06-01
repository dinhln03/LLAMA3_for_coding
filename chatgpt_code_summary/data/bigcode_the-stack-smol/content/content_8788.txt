def findDecision(obj): #obj[0]: Passanger, obj[1]: Coupon, obj[2]: Education, obj[3]: Occupation, obj[4]: Restaurant20to50, obj[5]: Distance
	# {"feature": "Passanger", "instances": 34, "metric_value": 0.9774, "depth": 1}
	if obj[0]<=1:
		# {"feature": "Restaurant20to50", "instances": 22, "metric_value": 0.7732, "depth": 2}
		if obj[4]<=3.0:
			# {"feature": "Education", "instances": 21, "metric_value": 0.7025, "depth": 3}
			if obj[2]<=2:
				# {"feature": "Occupation", "instances": 16, "metric_value": 0.8113, "depth": 4}
				if obj[3]<=10:
					# {"feature": "Coupon", "instances": 13, "metric_value": 0.6194, "depth": 5}
					if obj[1]>2:
						# {"feature": "Distance", "instances": 7, "metric_value": 0.8631, "depth": 6}
						if obj[5]<=2:
							return 'False'
						elif obj[5]>2:
							return 'False'
						else: return 'False'
					elif obj[1]<=2:
						return 'False'
					else: return 'False'
				elif obj[3]>10:
					# {"feature": "Coupon", "instances": 3, "metric_value": 0.9183, "depth": 5}
					if obj[1]<=2:
						return 'True'
					elif obj[1]>2:
						return 'False'
					else: return 'False'
				else: return 'True'
			elif obj[2]>2:
				return 'False'
			else: return 'False'
		elif obj[4]>3.0:
			return 'True'
		else: return 'True'
	elif obj[0]>1:
		# {"feature": "Coupon", "instances": 12, "metric_value": 0.8113, "depth": 2}
		if obj[1]>0:
			# {"feature": "Occupation", "instances": 11, "metric_value": 0.684, "depth": 3}
			if obj[3]<=20:
				# {"feature": "Restaurant20to50", "instances": 10, "metric_value": 0.469, "depth": 4}
				if obj[4]>1.0:
					return 'True'
				elif obj[4]<=1.0:
					# {"feature": "Education", "instances": 4, "metric_value": 0.8113, "depth": 5}
					if obj[2]<=1:
						return 'True'
					elif obj[2]>1:
						# {"feature": "Distance", "instances": 2, "metric_value": 1.0, "depth": 6}
						if obj[5]<=2:
							return 'True'
						else: return 'True'
					else: return 'True'
				else: return 'True'
			elif obj[3]>20:
				return 'False'
			else: return 'False'
		elif obj[1]<=0:
			return 'False'
		else: return 'False'
	else: return 'True'
