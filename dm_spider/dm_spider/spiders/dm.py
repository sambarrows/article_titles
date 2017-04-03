# -*- coding: utf-8 -*-
import re
from scrapy import Spider
from scrapy.http import Request

from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError
from datetime import datetime

class dmSpider(Spider):
	name = "dm"
	allowed_domains = ["www.dailymail.co.uk"]
	# start_urls = (
	# 	'http://www.dailymail.co.uk/news/article-4296818',
	# )

	def start_requests(self):
		url_base = 'http://www.dailymail.co.uk/news/article-'
		for i in range(4100000, 4200000):
			url = url_base + str(i)
			yield Request(url, callback=self.parse, errback=self.parse_error,  meta={'id': i})

	def parse_error(self, failure):
		if failure.check(HttpError):
			response = failure.value.response
			flag = 'HTTP_error'
		elif failure.check(DNSLookupError):
			response = failure.request
			flag = 'DNS_error'
		elif failure.check(TimeoutError):
			response = failure.request
			flag = 'Timeout_error'
		else:
			flag = 'Other_error'
			response = failure.request

		if hasattr(response, 'request'):
			article_id = response.request.meta['id']
		else:
			article_id = response.meta['id']

		result = dict(article_id=article_id, flag=flag, category='', title='',
			sponsored='', author='', published='', shares='', comments='', time=str(datetime.now()))
		return result

	def parse(self, response):
		article_id = response.meta['id']
		category  = re.findall('.co.uk/(\S*?)/', response.url)
		title     = response.xpath('//h1/text()').extract_first()
		sponsored = response.xpath('//*[@class="sponsored-by"]/text()').extract()
		author    = response.xpath('//*[@class="author-section byline-plain"]/a/text()').extract()
		published = response.xpath('//*[@class="article-timestamp article-timestamp-published"]/text()').extract()
		shares    = response.xpath('//*[@class="share-count"]/b/text()').extract()
		comments  = response.xpath('//*[@class="count-number"]/text()').extract()
		result = dict(article_id=article_id, flag='found', category=category, title=title,
			sponsored=sponsored, author=author, published=published, shares=shares, comments=comments, time=str(datetime.now()))
		return result
