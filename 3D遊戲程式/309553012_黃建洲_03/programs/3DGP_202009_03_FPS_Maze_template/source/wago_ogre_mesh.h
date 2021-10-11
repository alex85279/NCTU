/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
Date: Dec 2006 - Nov. 2020
Email: wingo.wong@gmail.com

// TERRAIN_INFO: Modified Editable terrain (gray image)

WAGO_OGRE_MESH: A class to create a map based on an image.
*/
#ifndef __WAGO_OGRE_MESH_OBJ_H__
#define __WAGO_OGRE_MESH_OBJ_H__
#include <vector>
#include "Ogre.h"
#include "OgreStringConverter.h"
#include "OgreException.h"
#include "game_obj_name_factory.h"
//#include "common.h"

using namespace Ogre;

using namespace std;
class GRAY_SCALE_IMAGE {
	float* mArray;  

	size_t mWidth, mHeight;

public:

	GRAY_SCALE_IMAGE();

	GRAY_SCALE_IMAGE(const float* array, size_t width, size_t height);

	GRAY_SCALE_IMAGE(const std::vector<float>& array, size_t width, size_t height);

	GRAY_SCALE_IMAGE(const GRAY_SCALE_IMAGE& g);
	~GRAY_SCALE_IMAGE();
	//
	size_t getWidth() const { return mWidth; }
	size_t getHeight() const { return mHeight; }
	const float *getImage() const { return mArray;}
	void set(const float* array, size_t width, size_t height);
	
/*
	float& atForModifying(int x, int y)
	{
		return mArray[x + y*mWidth];
	}
*/
	const float at(size_t x, size_t y) const
	{
		return mArray[x + y*mWidth];
	}
	const float atWithBoundCheck(size_t x, size_t y) const
	{
		if (x < 0) x = 0;
		else if (x >= mWidth) x = mWidth - 1;
		if (y < 0) y = 0;
		else if (y >= mHeight) y = mHeight - 1;

		return mArray[x + y*mWidth];
	}
	//
	const float at_right(size_t x, size_t y) const
	{
		x++;
		if (x >= mWidth) {
			x = mWidth-1;
		}
		return mArray[x + y*mWidth];
	}
	const float at_left(size_t x, size_t y) const
	{
		x--;
		if (x < 0) {
			x = 0;
		}
		return mArray[x + y*mWidth];
	}

		const float at_top(size_t x, size_t y) const
	{
		y++;
		if (y >= mHeight) {
			y = mHeight-1;
		}
		return mArray[x + y*mWidth];
	}
	const float at_bottom(size_t x, size_t y) const
	{
		y--;
		if (y < 0) {
			y = 0;
		}
		return mArray[x + y*mWidth];
	}

	//greater than v: return true
	bool gr(int x, int y, float v) const {
		if (x <0 ) return false;
		if (x >= (int)mWidth) return false;
		if (y <0 ) return false;
		if (y >= (int)mHeight) return false;
		return mArray[x + y*mWidth] > v;

	}
	void printf() {
		for (int j = 0; j < (int) mWidth; j++) {
			for (int i = 0; i < (int) mHeight; i++) {
				if (mArray[i + j*mWidth] > 0.5) {
					cout << "1";
				} else {
					cout << "0";
				}
			}
			cout << endl;
		}
	}

	void loadDynamicGrayScaleImage(float *iarray, int w, int h, bool autoDelete = true) {
		if (autoDelete) {
			if (mArray) {
				delete [] mArray;
			}
		}
		mArray = iarray;
		mWidth = w;
		mHeight = h;
	}
	void saveGrayScaleImageToImage(Image& image)
	{

		ushort* data = new ushort[mWidth*mHeight];

		for (size_t y = 0; y < mHeight; ++y)
			for (size_t x = 0; x < mWidth; ++x)
				data[y*mWidth + x] = ushort(at(x, y) * 0xffff);


		image.loadDynamicImage((uchar*)data, mWidth, mHeight, 1, PF_L16, true);
	}
	void dilate(int a_Amount, const GRAY_SCALE_IMAGE *a_Ref) {
		
		for (int j = 0; j < (int) mHeight; j++) {
			for (int i = 0; i < (int) mWidth; i++) {
				if (mArray[i + j*mWidth] > 0.5
					|| a_Ref->atWithBoundCheck(i, j) > 0.5 
					|| a_Ref->atWithBoundCheck(i, j+1) > 0.5
					|| a_Ref->atWithBoundCheck(i, j-1) > 0.5
					|| a_Ref->atWithBoundCheck(i+1, j) > 0.5
					|| a_Ref->atWithBoundCheck(i-1, j) > 0.5
					|| a_Ref->atWithBoundCheck(i+1, j+1) > 0.5
					|| a_Ref->atWithBoundCheck(i+1, j-1) > 0.5
					|| a_Ref->atWithBoundCheck(i-1, j+1) > 0.5
					|| a_Ref->atWithBoundCheck(i-1, j-1) > 0.5) 
					
				{
					mArray[i + j*mWidth] = 1.0;
				} else {
					mArray[i + j*mWidth] = 0;
				}
			}
		}
	}
};


class TERRAIN_INFO {
private:
	mutable Vector3 mNormAtPoint;
	float *mTerrainHeightMap;
	float w, h;
public:
	size_t nx, nz;
	float ox, oz;
	float dx, dz;
	
	AxisAlignedBox mAABB;
	

	TERRAIN_INFO(int nx, int nz, float w, float h, float ox, float oz, float dx, float dz) {
		mTerrainHeightMap = NULL;
		changeTo(nx, nz, w, h, ox, oz, dx, dz);
	}
	void changeTo(size_t nx, size_t nz, float w, float h, float ox, float oz, float dx, float dz) {
		if (mTerrainHeightMap) delete [] mTerrainHeightMap;
		this->nx = nx;
		this->nz = nz;
		this->w = w;
		this->h = h;
		this->ox = ox;
		this->oz = oz;
		this->dx = dx;
		this->dz = dz;
		mTerrainHeightMap = new float[nx*nz];
	}
	void updateAABB_y()
	{
		Vector3 min_v = mAABB.getMinimum();
		Vector3 max_v = mAABB.getMaximum();
		max_v.y = min_v.y = mTerrainHeightMap[0];
		for (size_t i = 1; i < nx*nz; i++) {
			if (min_v.y > mTerrainHeightMap[i]) {
				min_v.y = mTerrainHeightMap[i];
			} else if (max_v.y < mTerrainHeightMap[i]) {
				max_v.y = mTerrainHeightMap[i];
			}
		}
		mAABB.setMinimum(min_v);
		mAABB.setMaximum(max_v);
	}
	void splat(const float *brush, size_t _sx, size_t _sz, size_t _nx, size_t _nz, bool flg_pull = true, float strength = 1.0)
	{
		for (size_t j = 0; j < _nz; j++) {
			for (size_t i = 0; i < _nx; i++) {
				if (_sx + i >= 0 && _sx + i < nx
					&&
					_sz + j >= 0 && _sz + j < nz) {
						if (flg_pull) {
							addHeightAtVertex(_sx + i, _sz + j, brush[i+j*_nx]*strength);
						} else {
							addHeightAtVertex(_sx + i, _sz + j, -brush[i+j*_nx]*strength);
						}
					}
			}
		}
	}

	void flatten(const float *brush, size_t _sx, size_t _sz, size_t _nx, size_t _nz, bool flg_pull = true, float strength = 1.0)
	{
		float y = 0;
		int n = 0;
		if (flg_pull) {
		for (size_t j = 0; j < _nz; j++) {
			for (size_t i = 0; i < _nx; i++) {
				if (_sx + i >= 0 && _sx + i < nx
					&&
					_sz + j >= 0 && _sz + j < nz
					&&
					brush[i+j*_nx]!=0
					) {
						y += at(_sx + i, _sz + j);
						n++;
					}
			}
		}
		
		if ( n ==0 ) return;
		
		y /= n;
		}
		for (size_t j = 0; j < _nz; j++) {
			for (size_t i = 0; i < _nx; i++) {
				if (_sx + i >= 0 && _sx + i < nx
					&&
					_sz + j >= 0 && _sz + j < nz) {
						setHeightAtVertex(_sx + i, _sz + j, y);
					}
			}
		}
	}
	void setTerrainOrigin(float ox, float oz) {
		this->ox = ox;
		this->oz = oz;
	}

	void saveHeightMapToImage(Image& image)
	{
		float min_y = mAABB.getMinimum().y;
		float max_y = mAABB.getMaximum().y;
		float dy = max_y - min_y;

		ushort* data = new ushort[nx*nz];
		for (size_t z = 0; z < nz; ++z) {
			for (size_t x = 0; x < nx; ++x) {
				float v = (at(x, z) - min_y)/dy; // normalize to [0, 1]
				if (v < 0) v = 0;
				if (v > 1) v = 1;
				data[z*nx + x] = ushort(v * 0xffff); // convert into ushort format
			}
		}


		image.loadDynamicImage((uchar*)data, nx, nz, 1, PF_L16, true);
	}
	void setHeightAtVertex(size_t i, size_t j, float y) {
		mTerrainHeightMap[j*nx+i] = y;
	}

	void addHeightAtVertex(size_t i, size_t j, float dy) {
		mTerrainHeightMap[j*nx+i] += dy;
	}
	float at(size_t i, size_t j) const {
		return mTerrainHeightMap[j*nx+i];
	}
	float getHeightAtVertex(int i, int j) const {
		if (i<0) i = 0; 
		else if ((size_t)i>=nx) i = (int)(nx-1);
		if (j<0) j = 0; 
		else if ((size_t)j>=nz) j = (int)(nz-1);

		return mTerrainHeightMap[j*nx+i];
	}

	pair<int, int> getGridPoint(float x, float z) const {
		float ix = (x - ox)/dx;
		float jz = (z - oz)/dz;
		int i = (int) ix;
		int j = (int) jz;
		return make_pair(i, j);
	}

	float getHeightAt(float x, float z) const {
		float y = 0;
		float ix = (x - ox)/dx;
		float jz = (z - oz)/dz;
		int i = (int) ix;
		int j = (int) jz;
		float u = ix - i;
		float v = jz - j;
		float y0, y1, y2;
		bool flip = false;
		if (u+v>1) {
			u = 1-u;
			v = 1-v;
			flip = true;
			y0 = getHeightAtVertex(i+1, j+1);
			y1 = getHeightAtVertex(i, j+1);
			y2 = getHeightAtVertex(i+1, j);
		} else {
			y0 = getHeightAtVertex(i, j);
			y1 = getHeightAtVertex(i+1, j);
			y2 = getHeightAtVertex(i, j+1);
		}
		y = y0 + u*(y1-y0) + v*(y2-y0);
		return y;
	}

	const Vector3 &getNormalAtVertex(int i, int j) const {
		int ileft = i - 1;
		int jtop = j + 1;
		int flip = -1;
		if (ileft <0) {
			ileft = i+1;
			flip*=-1;
		}
		if (jtop >= nz) {
			jtop = j - 1;
			flip*=-1;
		}
		Vector3 a = Vector3(i, getHeightAtVertex(i, j), j);
		Vector3 b = Vector3(ileft, getHeightAtVertex(ileft, j), j);
		Vector3 c = Vector3(i, getHeightAtVertex(i, jtop), jtop);

		a.x *= i*w/(nx-1);
		b.x *= ileft*w/(nx-1);
		c.x *= i*w/(nx-1);
		//
		a.z *= j*h/(nz-1);
		b.z *= jtop*h/(nz-1);
		c.z *= j*h/(nz-1);

		mNormAtPoint = flip*(c-a).crossProduct(b-a);
		mNormAtPoint.normalise();
		return mNormAtPoint;
	}

	const Vector3 &getNormalAt(float x, float z) const {
		
		float ix = (x - ox)/dx;
		float jz = (z - oz)/dz;
		int i = (int) ix;
		int j = (int) jz;
		float u = ix - i;
		float v = jz - j;
		Vector3 y0, y1, y2;
		bool flip = false;
		if (u+v>1) {
			u = 1-u;
			v = 1-v;
			flip = true;
			y0 = getNormalAtVertex(i+1, j+1);
			y1 = getNormalAtVertex(i, j+1);
			y2 = getNormalAtVertex(i+1, j);
		} else {
			y0 = getNormalAtVertex(i, j);
			y1 = getNormalAtVertex(i+1, j);
			y2 = getNormalAtVertex(i, j+1);
		}
		mNormAtPoint = y0 + u*(y1-y0) + v*(y2-y0);
		mNormAtPoint.normalise();
		return mNormAtPoint;
	}

	pair<bool, Vector3> rayIntersects(const Ray& ray) const
	{

		Vector3 point = ray.getOrigin();
		Vector3 dir = ray.getDirection();

		if (!mAABB.contains(point))
		{
			pair<bool, Real> result = ray.intersects(mAABB);
			if (!result.first)
				return make_pair(false, Vector3::ZERO);

			point = ray.getPoint(result.second);
		}


		while (true)
		{
			float h = getHeightAt(point.x, point.z);
			if (point.y <= h)
			{
				point.y = h;
				return make_pair(true, point);
			}


			point += dir;

			if (point.x < mAABB.getMinimum().x 
				|| point.z < mAABB.getMinimum().z
				|| point.x > mAABB.getMaximum().x 
				|| point.z > mAABB.getMaximum().z)
				return make_pair(false, Vector3::ZERO);

		}
	}
};

typedef enum {
	WAGO_OGRE_OBJ_TYPE_LINE,
	WAGO_OGRE_OBJ_TYPE_TRIANGLE
} WAGO_OGRE_OBJ_TYPE;


class WAGO_OGRE_MESH_OBJ : protected GAME_OBJ_NAME_FACTORY
{
public:
	WAGO_OGRE_MESH_OBJ(SceneManager *m);
	~WAGO_OGRE_MESH_OBJ();
	void create();
    void create(const String &imageName);
	void animate(Real time_step);
	void setMaterial(const String &name);
	SceneNode* createTerrainFromGrayLevelImage(const GRAY_SCALE_IMAGE &b, float dx, float dz, float h);
	void translateSceneNode(float x, float y, float z);
	
	
private:
	WAGO_OGRE_OBJ_TYPE _type_tmp;
protected:

	MeshPtr mMesh;
	//
	float mOriginX, mOriginZ;
	GRAY_SCALE_IMAGE mGrayImage;
	GRAY_SCALE_IMAGE mMapImage;
	//

	AxisAlignedBox mAABB;
	SceneNode *mNode;
	Entity *mEntity;
	SceneManager *mSceneMgr;
	//
	HardwareVertexBufferSharedPtr posVertexBuffer;
	HardwareVertexBufferSharedPtr normalVertexBuffer;
	HardwareIndexBufferSharedPtr indexBuffer;
	HardwareVertexBufferSharedPtr texcoordsVertexBuffers;

	void buildMesh(HardwareVertexBufferSharedPtr posnormVertexBuffer);
	void createMesh(SceneNode *parent = NULL);
	void animate_Terrain(Real time_step);
	//
	SceneNode* createTerrainFromGrayLevelImage_Portion(SceneNode *parent, size_t sx, size_t sz, size_t nx, size_t nz, float dx, float dz, float size_x, float size_z);
	SceneNode* createTerrainFromGrayLevelImage_Entire(const GRAY_SCALE_IMAGE &b, float dx, float dz, float h);	
	void allocateLargeMemory();
protected:
	int mNumFaces;
	int mNumVertices;
	int mNumLines;
	unsigned short *mFaceIndices;
	float *_vertexBuf_old;
	float *_vertexBuf;
	float *_normalBuf;
	float *_txtBuf;
	String mMaterialName;
	float mDX;
	float mDZ;
	Image *mImage;
protected:
	TERRAIN_INFO *mTerrainInfo;
	void createDftModel_Wall_Horizontal();
	void createDftModel_Wall_Vertical();
	void createDftModel_Square();
	void createTerrain();
	//
	void createWallsBasedOnBitmap();
	void createWallsBasedOnBitmap(const String &imageName);

protected:
	void calculateVertexNormal_Terrain(size_t nx, size_t nz);
	void calculateVertexNormal_Terrain(int sx, int sz, size_t nx, size_t nz);
	void scaleTextureCoords(float fx, float fy);
	void translate(float x, float y, float z);
	void addFace(unsigned short i0, unsigned short i1, unsigned short i2);
	void bgnFace();
	void endFace();
	void addLine(unsigned short i0, unsigned short i1);
	void bgnLine();
	void endLine();
	void addVertex(float x, float y, float z, float txt_x, float txt_y, float nx = 0, float ny = 1, float nz = 0);
	void bgnVertex();
	void endVertex();
	const AxisAlignedBox &computeAABB();
};

class SIMPLE_TERRAIN : public WAGO_OGRE_MESH_OBJ {
protected:
	Vector3 mDestination;
	Vector3 mStartingPosition;
	std::vector<Vector3> mMonsterLocation;
    //
    Vector3* mNormalVecMap;             // NVM 
	size_t mNVM_Width, mNVM_Height;
    //
public:
	SIMPLE_TERRAIN(SceneManager *m);
	void dilateMapObstacles(int a_Amount);
    void computeNormalVectors( );
	void getDestination(Vector3 &d);
	void getStartingPosition(Vector3 &d);
	float getGridCellValue(float x, float z);
    Vector3 getGridNormalVector( float x, float z ) const;
	void scanMapForLocatingObjects();
	const std::vector<Vector3> &getLocationOfMonsters() const;
};

extern GRAY_SCALE_IMAGE convertImageToGrayScaleImage(const Image& image);
extern void convertImageToGrayScaleImage(const Image& image, float *image_array);
#endif